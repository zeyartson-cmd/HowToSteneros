# howtosteneros

An optimizer for training neural networks. It is AdamW whose main weights never move directly.
Instead, a **scout** runs ahead, and the main weights "slide" toward it only after the scout passes a test.

> **Status: small experiment (v0.1).** On small tasks it matches a well-tuned AdamW, but it is not better.
> The PyTorch port passes its tests (`python -m pytest tests`, 8/8, including a check that the scout step
> matches `torch.optim.AdamW`) and trains `examples/train_digits_torch.py` to about 96.6% on digits (one run).
> It has not been benchmarked against AdamW at scale.

---

# How the sliding works

There are two sets of weights:

- **a1** is the main weights. They change only after a successful test.
- **a2** is the scout (the target). It starts as a copy of a1 and runs ahead with plain AdamW.

One cycle:

1. a2 takes **k steps** of AdamW (usually k = 5). Gradients are computed at a2.
2. **Test:** take one fresh batch and compute the loss of a1 and of a2 on every example.
3. **If a2 is better** (statistically): a1 slides a fraction α toward a2, and the scout's learning rate goes up.
4. **If a2 is worse** (or the loss is NaN): a2 returns to a1, its memory (m and v) is zeroed, and the scout's learning rate goes down.

Bad steps are filtered out before they reach the main weights.

Related ideas: [Lookahead](https://arxiv.org/abs/1907.08610) (slow/fast weights, but no test) and
trust-region / line-search methods (accept or reject a step by the actual loss change).
This is a simple combination of those ideas, not a new class of optimizer.

---

# Formulas

## SGD

```
θ ← θ − lr · g
```

`g` is the minibatch gradient, `lr` is the learning rate.

## AdamW

```
m ← β1·m + (1 − β1)·g                    # momentum (smoothed gradient)
v ← β2·v + (1 − β2)·g²                   # scale (smoothed squared gradient)
m̂ = m / (1 − β1^t),   v̂ = v / (1 − β2^t)  # start-up bias correction
θ ← θ − lr · ( m̂ / (√v̂ + ε) + λ·θ )      # λ·θ is the weight decay
```

Typical values: β1 = 0.9, β2 = 0.999, ε = 1e-8, λ ≈ 0.01.

## HowToSteneros (ours)

Let θ₁ be the weights of a1, θ₂ the weights of a2 (the scout), and lr₂ the scout's learning rate.

**Step 1. The scout runs ahead** (k times): plain AdamW with its own learning rate.

```
θ₂ ← AdamW_step(θ₂, g(θ₂), lr₂)
```

**Step 2. Test** on one fresh batch of n examples:

```
dᵢ = Lᵢ(θ₁) − Lᵢ(θ₂)            # how much better the scout is on example i
d̄  = mean(dᵢ)
se = std(dᵢ) / √n                # standard error of the mean
accept  ⇔  d̄ > 0  and  d̄ > z · se
```

`z` is the significance threshold (z = 0 means just "better").

**Step 3. Slide:**

```
if accept:
    θ₁ ← θ₁ + α · (θ₂ − θ₁)             # a1 slides toward a2
    lr₂ ← min(lr₂ · up, lr_max)         # the scout gets bolder
else:
    θ₂ ← θ₁                             # the scout returns
    m, v ← 0                            # its memory is reset
    lr₂ ← max(lr₂ · down, lr_min)       # the scout gets careful
```

**In one line:**

```
θ₁ ← θ₁ + α · [accept] · (θ₂ − θ₁),   where θ₂ is θ₁ after k steps of AdamW
```

Typical values: α = 0.5, up = 1.1, down = 0.5, k = 5, z = 0…1.

Optional: `lr_max` / `anneal_steps` (an lr ceiling, optionally cosine-decayed) and
`reset_on_accept` (Lookahead-style: after a success the scout restarts from a1).

## How it differs from AdamW

| | AdamW | HowToSteneros |
|---|---|---|
| What moves the main weights | every gradient step | only a scout that passed the test |
| Learning rate | follows a schedule | adapts by itself (up / down) |
| A bad step | goes into the weights | is discarded |
| Cost | 1 set of weights + 2 numbers per weight | an extra copy of the weights and 2 forward passes on the test batch every k steps |

---

# Honest results

Small MLPs (NumPy), equal budget of gradient evaluations, hyperparameters of **both** methods tuned
separately on tuning seeds by validation loss, final numbers on 10 fresh test seeds (mean ± standard error).
Tasks: scikit-learn `digits`, and a noisy synthetic 6-class problem (15% label noise).

| Task | Metric | Tuned AdamW | HowToSteneros |
|---|---|---|---|
| digits | test accuracy | 97.44 ± 0.20 % | 97.40 ± 0.22 % |
| digits | test loss | 0.103 ± 0.012 | 0.116 ± 0.013 (slightly worse) |
| noisy | test accuracy | 48.45 ± 0.46 % | 48.63 ± 0.52 % |
| noisy | test loss | 1.463 ± 0.015 | 1.425 ± 0.011 (slightly better) |

**Takeaway: with a tuned learning rate, it is on par with AdamW, not better.**

### Bad starting learning rate (ablation)

Start lr = 0.1, far too high for these nets.

| Task | AdamW | AdamW, lr clamped to the cap | HowToSteneros, with lr cap | HowToSteneros, **no cap** |
|---|---|---|---|---|
| digits acc | 53.8 ± 3.6 % | 97.6 ± 0.2 % | 97.4 ± 0.2 % | 96.8 ± 0.2 % |
| noisy acc | 20.0 ± 1.7 % | 49.6 ± 0.4 % | 48.5 ± 0.5 % | 36.7 ± 2.4 % |

- The test gives real protection: **without any cap** it recovers from lr = 0.1
  (digits 96.8 % vs 53.8 % for plain AdamW), because failed scouts are discarded and the scout lr shrinks.
- But much of the benefit is just an lr ceiling: AdamW clamped to the same ceiling does as well.
  On the noisy task, recovery without a cap is only partial (36.7 % vs about 49 %).

### What did not help

A third path `b1` that splits into two branches with different learning rates and picks the best of three
(equal gradient budget): no gain in accuracy and worse recovery from a bad lr. It is not included.

### What is not shown

Large models, transformers, long training, real-world datasets, memory/time measurements.

---

# Usage

```bash
pip install -e .                 # NumPy core
pip install -e ".[torch]"        # + PyTorch
python benchmarks/bench2.py      # full comparison (about 20-30 min on 2 cores)
python benchmarks/ablation_cap.py 0.1   # the bad-lr ablation above
python -m pytest tests           # NumPy tests (torch tests are skipped without torch)
```

```python
import torch.nn.functional as F
from howtosteneros import HowToSteneros

opt = HowToSteneros(model.parameters(), lr=1e-3, k=5, z=0.5)

for x, y in loader:
    opt.zero_grad(); loss_fn(model(x), y).backward()
    opt.step()                       # scout step (AdamW)
    if opt.ready():                  # every k steps
        xt, yt = next(test_iter)     # a FRESH batch, same one for a1 and a2
        opt.judge(lambda: F.cross_entropy(model(xt), yt, reduction="none"))

opt.load_main()                      # put the main weights (a1) into the model
```

Limitations of the PyTorch port: BatchNorm buffers are not tracked; the `judge` callable must return
per-example losses and should run the model in eval mode.

The NumPy version is `from howtosteneros import HowToStenerosNumpy` (see `benchmarks/bench2.py`).

---

# License

Apache License 2.0. See [LICENSE](LICENSE).
