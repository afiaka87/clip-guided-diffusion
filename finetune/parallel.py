"""One run on several GPUs: a process per GPU, their gradients averaged at every step.

    uv run --no-sync python -m finetune.parallel --devices 0,1,2,3 finetune.cond --hours 12 --wandb

starts `finetune.cond` once per device with the options after its name, each process seeing only its own GPU, and
stops them all as soon as one fails, so a crash never leaves the rest waiting for it. `--devices cpu,cpu` runs
on the CPU, and a device may be named twice.

A script takes part with four calls. `join()` says which process this is, `shared()` wraps the model so that
every backward pass averages the gradients of all of them, `agreed()` is a decision the first process makes for
all, and `averaged()` is a record's mean over them. Every process then holds the same weights at every step, as
long as each applies the same update to them, so only the first one has to log, check and save. Without the
launcher all four leave a script as it was on one GPU.

Linux only: PyTorch's builds for Windows cannot connect processes (`unsupported gloo device`).
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import timedelta
from typing import NamedTuple

import torch as th
from torch import distributed as dist
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel

RANK, SIZE, PORT, BACKEND = "CGD_TEAM_RANK", "CGD_TEAM_SIZE", "CGD_TEAM_PORT", "CGD_TEAM_BACKEND"
POLL_SECONDS = 0.5
# How long the others wait for the first process to draw samples and save, which NCCL would cut off at 10 minutes.
# A crash ends every process sooner than this: the launcher stops the rest as soon as one exits.
WAIT = timedelta(minutes=60)
GRACE_SECONDS = 20


class Team(NamedTuple):
    """Which of a run's processes this one is. `linked` is false for a run that was started by itself."""

    rank: int = 0
    size: int = 1
    linked: bool = False
    device: str = "cpu"

    @property
    def lead(self) -> bool:
        """Whether this is the process that logs, checks and saves."""
        return self.rank == 0


def join(device: str) -> Team:
    """This process's place among those the launcher started, connected to the rest. Alone without the launcher."""
    if SIZE not in os.environ:
        return Team(device=device)
    rank, size = int(os.environ[RANK]), int(os.environ[SIZE])
    address = f"tcp://127.0.0.1:{os.environ[PORT]}"
    # Its own GPU by name, which NCCL otherwise guesses from the rank: each process sees one GPU, number 0.
    place = th.device(device)
    own = th.device("cuda", place.index or 0) if place.type == "cuda" else None
    dist.init_process_group(
        os.environ[BACKEND], init_method=address, rank=rank, world_size=size, device_id=own, timeout=WAIT
    )
    return Team(rank, size, True, device)


def leave(team: Team) -> None:
    """Disconnect from the other processes."""
    if team.linked:
        dist.destroy_process_group()


def shared(team: Team, model: nn.Module) -> nn.Module:
    """The model to take the loss through: every backward pass through it averages all processes' gradients.

    It also copies the first process's weights to the rest, in place, so they start from the same ones. On GPUs
    the gradients travel as bfloat16, half the bytes. On four RTX 5090s without a direct link between them, which
    send them through main memory, that took a step of 6 images each from 25.9 images a second to 35.5.
    """
    if not team.linked:
        return model
    device = th.device(team.device)
    if device.type != "cuda":
        return DistributedDataParallel(model)
    net = DistributedDataParallel(model, device_ids=[device.index or 0])
    net.register_comm_hook(None, default_hooks.bf16_compress_hook)
    return net


def quiet(net: nn.Module, sending: bool) -> AbstractContextManager:
    """Where a batch whose gradient is only added to, not yet sent, runs: `sending` is false for those."""
    return nullcontext() if sending or not isinstance(net, DistributedDataParallel) else net.no_sync()


def unshared(model: nn.Module) -> nn.Module:
    """The model itself, from `shared`'s result."""
    return model.module if isinstance(model, DistributedDataParallel) else model


@contextmanager
def lead_first(team: Team) -> Iterator[None]:
    """Run a block in the first process before the others start it, for what downloads or caches on disk.

    Nothing inside may be shared between the processes, such as `shared()`: the others are waiting, not answering.
    """
    if team.linked and not team.lead:
        dist.barrier()
    yield
    if team.linked and team.lead:
        dist.barrier()


def agreed(team: Team, decision: bool) -> bool:
    """The first process's decision, in every process. Each must call it at the same step."""
    if not team.linked:
        return decision
    flag = th.tensor([int(decision)], device=team.device)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def averaged(team: Team, record: Mapping[str, float]) -> dict[str, float]:
    """A record's mean over the processes. Each must call it at the same step, with the same names."""
    if not team.linked or not record:
        return dict(record)
    values = th.tensor(list(record.values()), dtype=th.float64, device=team.device)
    dist.all_reduce(values)
    return dict(zip(record, (values / team.size).tolist(), strict=True))


def free_port() -> int:
    """A port nothing listens on."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def default_backend(devices: Sequence[str]) -> str:
    """NCCL between GPUs on Linux, which is the fast one. Gloo anywhere else: on Windows, or with a CPU among them."""
    return "nccl" if sys.platform == "linux" and "cpu" not in devices else "gloo"


def stop(processes: Sequence[subprocess.Popen]) -> None:
    """End every process that is still running, by force if it does not go when asked."""
    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()
    deadline = time.monotonic() + GRACE_SECONDS
    for process in running:
        try:
            process.wait(max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()


def launch(devices: Sequence[str], module: str, arguments: Sequence[str], backend: str | None = None) -> int:
    """Run `module` once per device and wait. Returns 0 when all finished, or the first failure's exit code."""
    team = {SIZE: str(len(devices)), PORT: str(free_port()), BACKEND: backend or default_backend(devices)}
    processes = []
    try:
        for rank, device in enumerate(devices):
            seen = {"CUDA_VISIBLE_DEVICES": "" if device == "cpu" else device}
            environment = os.environ | team | seen | {RANK: str(rank)}
            command = [sys.executable, "-m", module, *arguments]
            # pylint: disable-next=consider-using-with  # They run together, and `stop` ends each one.
            processes.append(subprocess.Popen(command, env=environment))
        while True:
            codes = [process.poll() for process in processes]
            failed = [code for code in codes if code]
            if failed:
                print(f"A process ended with exit code {failed[0]}. Stopping the rest.", file=sys.stderr, flush=True)
                return failed[0]
            if all(code == 0 for code in codes):
                return 0
            time.sleep(POLL_SECONDS)
    finally:
        stop(processes)


def interrupted(_number: int, _frame: object) -> None:
    """Turn a request to end into the exception that makes `launch` stop its processes on the way out."""
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    """The command: the devices, a module's name, and that module's own options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--devices", required=True, help="GPU numbers with commas between, such as 0,1,2,3, or cpu")
    parser.add_argument("--backend", choices=["nccl", "gloo"], help="How the processes talk. Chosen by default.")
    parser.add_argument("module", help="The script to run in each process, such as finetune.cond")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Its options")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, interrupted)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupted)
    return launch(args.devices.split(","), args.module, args.arguments, args.backend)


if __name__ == "__main__":
    sys.exit(main())
