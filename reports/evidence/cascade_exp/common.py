import gc, json, os, subprocess, sys, time
REPO = "/home/sam/GitHub/clip-guided-diffusion"
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
os.chdir(HERE)
import torch as th
PROMPTS = [
    "a red fox in the snow", "a plate of sushi on a black slate",
    "a pencil sketch of an old man with a beard", "a lighthouse on a rocky coast at sunset",
    "a bowl of ramen with a soft boiled egg", "an oil painting of a sailing ship in a storm",
    "a golden retriever puppy on a wooden floor", "a steam locomotive crossing a stone bridge",
    "a watercolor of cherry blossoms by a river", "a red vintage sports car on a desert road",
    "a snow covered cabin in a pine forest at night", "a macro photograph of a dew covered dragonfly",
]
LOG = open(os.path.join(HERE, "results.jsonl"), "a", buffering=1)

def others_mib():
    used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    return used - th.cuda.memory_reserved() / 2**20

def gate(limit=1500, patience=3):
    start = time.time()
    while others_mib() > limit and time.time() - start < patience:
        time.sleep(3)

def measured(fn, retries=30):
    for _ in range(retries):
        gate()
        before = others_mib()
        th.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            result = fn()
        except th.cuda.OutOfMemoryError:
            gc.collect(); th.cuda.empty_cache(); time.sleep(20)
            continue
        th.cuda.synchronize()
        return result, dict(sec=round(time.perf_counter() - start, 3), peak_alloc_gib=round(th.cuda.max_memory_allocated() / 2**30, 3),
                            peak_reserved_gib=round(th.cuda.max_memory_reserved() / 2**30, 3), others_before=round(before), others_after=round(others_mib()))
    raise RuntimeError("gave up")

def record(pipe, prompt, path, stats):
    LOG.write(json.dumps(dict(pipe=pipe, prompt=prompt, path=path, **stats)) + "\n")

def paths(name):
    return {r["prompt"]: r["path"] for r in map(json.loads, open(os.path.join(HERE, "results.jsonl"))) if r.get("pipe") == name}
