"""Is the repo's tv_loss exposed to tf.pad's nondeterministic backward?

tf.pad replicate differs at every width when the adjoint on the padded pixels is random.
tv_loss pads the right and bottom by one, but a replicated pixel minus its source is
exactly zero, so the squared difference sends exactly zero back. Check that, using the
repo's own function, then with the same pad under a loss that does send gradient back.
"""
import sys

import torch as th
import torch.nn.functional as tf

sys.path.insert(0, "/home/sam/GitHub/clip-guided-diffusion")
from cgd.losses import tv_loss  # noqa: E402

device = "cuda"
REPEATS = 200
th.manual_seed(0)
x0 = (0.5 + 0.2 * th.randn(1, 3, 128, 128, device=device)).clamp(0, 1)


def check(name, fn):
    base, differing, worst, where = None, 0, 0.0, None
    for _ in range(REPEATS):
        x = x0.clone().requires_grad_(True)
        g = th.autograd.grad(fn(x).sum(), x)[0].detach()
        if base is None:
            base = g.clone()
        elif not th.equal(g, base):
            differing += 1
            delta = (g - base).abs()
            worst = max(worst, delta.max().item())
            where = (delta > 0).nonzero()[:4].tolist()
    print(f"{name:52s} {'identical' if not differing else f'DIFFERS {differing}/{REPEATS - 1} max {worst:.2e} at {where}'}", flush=True)


w = th.randn(1, 3, 129, 129, device=device)
check("repo tv_loss", tv_loss)
check("tf.pad right+bottom 1, random adjoint", lambda v: (tf.pad(v, (0, 1, 0, 1), "replicate") * w).sum((1, 2, 3)))
