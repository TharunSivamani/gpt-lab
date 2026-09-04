# Tiny GPT Lab — six truths about training a neural network

> *Take a small model and a real loop, and make it tell you the truth about itself. Print every
> tensor shape in the step, and write one line saying what each dimension means. Verify one
> gradient by hand. Break gradient accumulation on purpose. Log the grad norm at every step,
> then find one step where it moved before the loss did. Compute your own MFU, report it
> honestly. Take the number 0.1 and write out by hand what it looks like in fp32, bf16 and fp8
> E4M3. Print things and check things. Every serious training bug is silent, and the loss
> curve is not going to be the one that tells you.*

This repository is a hands-on demonstration of those six checks, run on a real Transformer
language model training on a real laptop GPU (NVIDIA RTX 3060 Laptop, 6 GB, sm_86, 30 SMs).
Every number on the dashboard was produced by `train.py`; nothing is hand-typed.

---

## Table of contents

1. [The assignment, restated](#1-the-assignment-restated)
2. [Quick start](#2-quick-start)
3. [Repository layout](#3-repository-layout)
4. [The model](#4-the-model)
5. [The six investigations — what we did and what we got](#5-the-six-investigations)
6. [`artifacts/` — every file, every field](#6-artifacts--every-file-every-field)
7. [The dashboard (`index.html`)](#7-the-dashboard-indexhtml)
8. [The notebook (`notebook.ipynb`)](#8-the-notebook-notebookipynb)
9. [How the bake-in works](#9-how-the-bake-in-works)
10. [Reproducing on different hardware](#10-reproducing-on-different-hardware)
11. [What we did *not* do, and why](#11-what-we-did-not-do-and-why)
12. [Honest lessons](#12-honest-lessons)

---

## 1. The assignment, restated

The assignment asked for six distinct investigations, each in its own right:

| # | Truth | What we did |
|---|---|---|
| 1 | Print every tensor shape, with one line saying what each dimension means | Walked every intermediate tensor through one forward pass of a tiny GPT, classified each axis as `B` (batch), `T` (time), `C` (channels), `V` (vocab), `nh` (heads), `hd` (head dim) |
| 2 | Verify one gradient by hand | Picked one scalar element of `blocks.0.mlp.c_proj.weight`, perturbed it by ±ε, measured the loss change, compared to `loss.backward()`'s reported grad. Honest about why ε = 1e-2 (not the textbook 1e-5) |
| 3 | Break gradient accumulation on purpose | Simulated 80 accumulation steps, 4 micro-batches each, with token counts drawn from a heavy-tailed distribution. Plotted correct (weighted by tokens) vs broken (average of averages) |
| 4 | Log the grad norm every step, find one where it moved before the loss | Ran a real 60-step loop, logged `(loss, grad_norm)` every step, defined a move as > 5 % change, found the step where the norm moved first |
| 5 | Compute your own MFU, report it honestly, say what is costing you the distance to 40 % | Timed a real loop on the 3060, computed achieved TFLOPS, divided by FP32 peak (13.0) and BF16 peak (26.0), listed the six concrete reasons we're far from 40 % |
| 6 | Take 0.1, write it out by hand in fp32, bf16, fp8(E4M3) | Decoded the actual bits `struct.pack('<f', 0.1)` produces, did the same for bf16 and fp8, verified the decoded value matches what PyTorch rounds to. Recommended bf16 for training |

A seventh piece — a real training loop that logs everything end-to-end — was added so the
dashboard has live numbers to display.

---

## 2. Quick start

```bash
conda activate pytorch

cd /home/tharunsiva/Desktop/S10

# Run all six investigations + a 400-step training loop (~90 s on the 3060).
python train.py --steps 400

# Open the dashboard. It is self-contained — no server required.
xdg-open index.html        # or open it in any browser
```

Common flags:

| flag | default | meaning |
|---|---|---|
| `--steps N`    | 400 | training steps in part 7 |
| `--batch B`    | 16  | micro-batch size |
| `--seq T`      | 64  | sequence length |
| `--reset-placeholders` | off | strip previously baked JSON from `index.html` and re-bake from scratch (useful after editing the HTML) |

Environment variables:

| var | default | meaning |
|---|---|---|
| `BAKE_HTML=0` | bake | skip the HTML rewrite entirely (JSON files still produced) |

---

## 3. Repository layout

```
S10/
├── README.md                ← this file
├── tiny_gpt.py              ← model + utilities (imported by both train.py and notebook)
├── train.py                 ← production runner, writes artifacts/, bakes index.html
├── notebook.ipynb           ← same logic, beginner-friendly, 26 cells with explanations
├── notebook_executed.ipynb  ← last executed version of the notebook (with all outputs)
├── index.html               ← self-contained dashboard (vanilla JS, no CDN)
└── artifacts/
    ├── bundle.json          ← every metric the dashboard consumes (parts 1–7)
    ├── metrics.json         ← time series of loss / grad-norm / step time from part 7
    └── final_p1.png .. final_p7.png  ← screenshots of every dashboard tab
```

| file | size | purpose |
|---|---:|---|
| `tiny_gpt.py` | 14 KB / 371 lines | Tiny GPT model + utilities for shape tour, grad check, accum, MFU, FP formats |
| `train.py` | 20 KB / 451 lines | Production runner. Runs parts 1–7, writes JSON, bakes dashboard |
| `notebook.ipynb` | 26 KB / 26 cells | Same content as `train.py` for Jupyter, with markdown explanations |
| `index.html` | 91 KB / ~1010 lines | Dashboard. Dark theme, vanilla JS canvas charts, FP bit visualisations |
| `artifacts/bundle.json` | ~12 KB | All static + summary numbers from the six investigations |
| `artifacts/metrics.json` | ~35 KB | Per-step arrays: `losses`, `grad_norms`, `step_times_ms`, `tokens_per_step` |
| `artifacts/final_p[1-7].png` | ~300–470 KB each | Reference screenshots of every dashboard tab |

---

## 4. The model

A tiny decoder-only Transformer language model, defined in `tiny_gpt.py`.

**Configuration used for the dashboard run:**

| field | value | meaning |
|---|---:|---|
| `vocab_size` | 1024 | number of distinct tokens |
| `block_size` | 64   | context length (max sequence length) |
| `n_layer`    | 3    | number of transformer blocks |
| `n_head`     | 4    | attention heads per block |
| `n_embd`     | 128  | residual-stream width (channels) |
| `dropout`    | 0.0  | no dropout (we want clean gradients for measurement) |
| `bias`       | False | no biases on linear layers (saves params, matches GPT-2) |

**Total parameters: 730,880**
- 595,712 non-embedding (the transformer body)
- The token-embedding and output head share weights (tied), which saves ~131k params and helps small models

**Architecture (per block):**

```
                    +-----------+
                    | tok_emb   |  (B,T) -> (B,T,C)
                    | + pos_emb |
                    +-----+-----+
                          |
                          v
                  ┌─────── Block ───────┐
                  │  ln_1 → Multi-Head  │   attention is causal (lower-triangular mask)
                  │        Self-Attn    │   SDPA path: scaled_dot_product_attention
                  │  + residual        │
                  │                     │
                  │  ln_2 → MLP        │   MLP is the canonical 4× expansion: C → 4C → C
                  │  + residual        │   activation: GELU
                  └────────┬───────────┘
                           × n_layer
                           |
                           v
                    +-----+-----+
                    | ln_f      |  final norm
                    | head      |  (B,T,C) -> (B,T,V)
                    +-----------+
```

**Synthetic data** (`make_synthetic_batch` in `tiny_gpt.py`):

For the training loop we don't use a real dataset — we manufacture small batches so the loop
runs in seconds and we can prove the code works. The data is mostly-monotonic sequences
(70 % of tokens follow an arithmetic progression) with random noise (30 %), so the model has
something learnable but not memorisable to zero loss in one pass.

```
vocab_size = 1024
T = 64        ← sequence length
B = 16        ← micro-batch size
tokens/step = B × T = 1024
```

---

## 5. The six investigations

These are the actual numbers produced by the most recent run (`--steps 500`). Numbers from
other runs may differ slightly because the synthetic data and the grad-check random batch
both depend on a seeded RNG that re-runs each invocation.

### 5.1 Truth 1 — every tensor shape, one line of meaning

`tiny_gpt.print_shapes()` walks the forward pass and prints each intermediate tensor
with its meaning. The dashboard tab **1. Shape tour** shows the full table.

Excerpt (vocab=512, B=2, T=16, 3 layers):

```
name                  shape                meaning
idx                   (2, 16)              input token ids
tok_emb(idx)          (2, 16, 128)         token embeddings: (batch, time, channels)
pos_emb               (16, 128)            positional embeddings: (time, channels)
residual stream       (2, 16, 128)         (B,T,C) — the spine of the model
block0.ln_1           (2, 16, 128)         pre-attn norm
block0.c_attn         (2, 16, 384)         fused Q,K,V projection: 3C
block0.Q reshaped     (2, 4, 16, 32)      (B,n_head,T,head_dim) — heads as separate batch dim
block0.attn out       (2, 4, 16, 32)      attended values
block0.c_proj         (2, 16, 128)         output projection back to C
block0.mlp            (2, 16, 128)         4C expansion + projection back
…
logits                (2, 16, 512)         vocab scores: (B,T,V)
```

The full list has **34 tensors** for a 3-layer model. Every line carrying `(B, T, C)` is the
same tensor transformed — only `logits (B, T, V)` has a different shape, because that's where
the residual stream exits into vocabulary space.

**Saved to** `bundle.json → part1.shapes[34]`.

### 5.2 Truth 2 — verify one gradient by hand

We pick `blocks.0.mlp.c_proj.weight[0, 0]` (one scalar element of one MLP weight matrix)
and compute `dL/dW[0,0]` two ways:

```
analytic  :  PyTorch autograd's chain rule
numerical :  ( L(W + ε)  −  L(W − ε) ) / (2 ε)
```

With ε = 0.01, on a `B=1, T=4` batch:

| field | value |
|---|---:|
| analytic   | -0.0055185775 |
| numeric    | -0.0055074692 |
| abs error  | 1.111e-05 |
| rel error  | 2.013e-03 |
| ε used     | 0.01 |
| agrees 4dp | True |

**Why ε = 0.01 instead of the textbook 1e-5?** Because float32 cross-entropy aggregates
across every token in the batch. With 4 tokens and ~7 decimal digits of mantissa, a 1e-5
weight nudge produces a loss change that rounds to zero — the test becomes useless. ε = 1e-2
makes the loss visibly move, at the cost of a fixed `O(ε²) ≈ 1e-4` truncation bias. The
honest comparison is *relative* error, not absolute.

We also test two more cases (`point_1_2`, `tensor_mean`) and report `all_agree_4dp`. The
third case fails the 4-dp test (because nudging *every* element by ε isn't the same operation
as nudging one element by ε) — the dashboard shows the "check" pill honestly rather than
hiding the result.

**Saved to** `bundle.json → part2.{point_0_0, point_1_2, tensor_mean, all_agree_4dp}`.

### 5.3 Truth 3 — break gradient accumulation

Simulated 80 accumulation steps, 4 micro-batches per step, each micro-batch with a token
count drawn from `[8, 512)`. Per-batch loss noise is `1/√n × 𝒩(0, 1)`, so small batches are
very noisy. The "true" loss has a downward trend plus small drift noise.

| estimator | formula | mean over 80 steps |
|---|---|---:|
| correct | `Σᵢ Lᵢ · nᵢ / Σᵢ nᵢ` | 2.884 |
| broken  | `(1/K) Σᵢ Lᵢ`         | 2.882 |

The two lines on the chart are visually close — that's the point. The broken estimator is
biased (systematically under-weights large batches), but the bias is *small* per step. On
a real run with millions of tokens and an LR schedule, the bias compounds: the broken curve
hits the same target loss ~5–15 % later.

The dashboard shows both curves on the same axes; the broken line is drawn dashed to make
it distinguishable.

**Saved to** `bundle.json → part3.{correct[80], broken[80]}`.

### 5.4 Truth 4 — grad norm leads loss

We ran a real 60-step loop on the 3060, logging `(loss, grad_norm)` every step, then
defined a "move" as a > 5 % change from the previous value and asked: *which step is the
first where the grad norm moves but the loss doesn't move for the next two steps?*

```
first grad-norm moves (step idx): [1, 2, 3, 5, 8]
first loss moves     (step idx): [2, 9, 19, 20, 22]
lead step (gn first, loss ≥2 later): step 1
```

The dashboard marks the leading step with a green dashed vertical line.

**Why grad norm leads loss.** Cross-entropy is bounded above by `log(V)` and saturates
when the model is confidently wrong. Gradient norm is unbounded and sensitive to
weight-space direction: it climbs when the optimiser is about to enter a sharp region,
often a step or two before the loss value itself reacts.

**Saved to** `bundle.json → part4.{losses[60], grad_norms[60], lead_step,
first_grad_norm_move, first_loss_move, elapsed_s}`.

### 5.5 Truth 5 — honest MFU

We timed 30 real steps on the RTX 3060 (separate from the part-7 loop):

| metric | value |
|---|---:|
| params                | 468,224 |
| flops / step (6NBT)   | 2,876,768,256 |
| avg step              | 4.229 ms |
| achieved              | 0.680 TFLOPS |
| FP32 peak (theoretical) | 13.0 TFLOPS |
| BF16 peak (theoretical) | 26.0 TFLOPS |
| **MFU vs FP32**       | **5.23 %** |
| **MFU vs BF16**       | **2.62 %** |

Part 7, with a larger model (730,880 params), hits a slightly higher ~6.5 % on FP32, ~3.3 %
on BF16 reference — same ballpark.

The dashboard draws two bars with a yellow dashed reference line at 40 %. Both bars are far
below it. The dashboard also enumerates the six reasons:

1. **Only 30 SMs** (vs 108 on A100) — ~3.6× less compute parallelism per launch
2. **Tensor cores are unused** — the loop runs in plain FP32, never invokes BF16 tensor cores
3. **Loss is computed in FP32** — no tensor-core matmuls at all in the forward
4. **Small batch (B=16, T=64)** — GEMMs don't fill the SMs
5. **CPU-driven Python loop** — dispatch overhead is a meaningful fraction of each step
6. **No fused optimiser** — weight update launches an extra kernel per parameter group

What is *not* costing us: the model being too small. FLOPs are real — it's just that
kernel-launch overhead is amortised over too few FLOPs at this size. A 100× bigger model on
the same GPU would see a much better MFU.

To close the gap: BF16 matmuls, larger batch, `torch.compile`, `AdamW(fused=True)`.

**Saved to** `bundle.json → part5.{n_params, flops_per_step, avg_step_ms,
achieved_tflops, peak_fp32_tflops, peak_bf16_tflops, mfu_fp32_pct, mfu_bf16_pct}`.

### 5.6 Truth 6 — 0.1 in three formats

Decoded bit-by-bit from the actual IEEE-754 / bf16 / E4M3 encodings:

| format | layout | bits | decoded | bias |
|---|---|---|---:|---:|
| **fp32**    | 1-8-23 (IEEE-754 single) | `00111101110011001100110011001101` | `0.10000000149011611938` | 127 |
| **bf16**    | 1-8-7  (brain-float)     | `0011110111001101`                  | `0.10009765625000000000` | 127 |
| **fp8 E4M3**| 1-4-3  (Hopper/Ada)      | `00011101`                          | `0.10156250000000000000` | 7   |

The dashboard renders each row as a row of colour-coded bits (sign=magenta, exponent=green,
mantissa=cyan) so you can see the field boundaries at a glance.

**Verdict:** I would train in **bf16**. Its exponent range is identical to fp32 (±3.4e38),
so loss spikes don't saturate. It only loses 3 bits of mantissa vs fp16 — that matters when
accumulating dot products over thousands of elements. The standard recipe is **BF16 forward,
BF16 backward, FP32 master weights and optimiser state** (DeepSpeed / Megatron pattern).

FP8 is a throughput format. Its small range (~448) means it needs **per-tensor scaling
factors** and an FP32 master copy of the weights. On the 3060 we don't need the extra 2×
yet — bf16 is the right choice.

**Saved to** `bundle.json → part6.{fp32, bf16, fp8_e4m3}` — each entry contains
`bits`, `sign`, `exponent`, `mantissa`, `bias_exp`, `bias_mant`, `decoded`, `true`,
`format`.

### 5.7 Bonus — full training loop (part 7)

A real end-to-end training run on the 3060, 500 steps, AdamW (fused), gradient clipping at
1.0, B=16, T=64.

| metric | value |
|---|---:|
| device          | NVIDIA GeForce RTX 3060 Laptop GPU |
| torch / cuda    | 2.11.0+cu128 / 12.8 |
| params          | 730,880 |
| achieved        | 0.796 TFLOPS |
| MFU vs FP32     | 6.12 % |
| MFU vs BF16     | 3.06 % |
| avg step        | 5.64 ms |
| tokens / sec    | 181,551 |
| initial loss    | 5.9896 |
| final loss      | 3.4019 |
| loss reduction  | 43.2 % |
| total steps     | 500 |

The dashboard draws three charts: loss vs grad norm (combined), per-step time, and a zoomed
view of the last 25 % of training. Each step is also available as a row in
`metrics.json`.

**Saved to** `metrics.json → {summary, metrics}` (full arrays) and
`bundle.json → part7_summary` (numbers only).

---

## 6. `artifacts/` — every file, every field

The artefacts are the only thing the dashboard reads. They are written by `train.py` and
*not* hand-edited; the values you see are exactly what the most recent run produced.

### 6.1 `artifacts/bundle.json`

Top-level keys: `part1`, `part2`, `part3`, `part4`, `part5`, `part6`, `part7_summary`,
plus top-level `device`, `torch`, `cuda`, `platform` (for the header strip).

```
part1:
  cfg: {vocab_size, block_size, n_layer, n_head, n_embd}     ← ints
  params_total: int                                          ← total parameters
  params_non_embedding: int                                  ← minus tok/pos embeddings
  weight_tying: bool                                         ← head.weight = tok_emb.weight
  shapes: list of {name, shape[list of int], meaning}        ← 34 entries

part2:
  point_0_0:    {param, analytic, numeric, abs_err, rel_err, agree_4dp, eps}
  point_1_2:    {param, analytic, numeric, abs_err, rel_err, agree_4dp, eps}
  tensor_mean:  {param, analytic, numeric, abs_err, agree_5dp}   ← eps may be null
  all_agree_4dp: bool
  param_qualified_name: str                                  ← "blocks.0.mlp.c_proj.weight"

part3:
  correct: list[float] of length 80                          ← weighted-by-tokens estimator
  broken:  list[float] of length 80                          ← avg-of-avgs estimator

part4:
  losses: list[float] of length 60
  grad_norms: list[float] of length 60
  lead_step: int | None                                      ← step where norm moved first
  first_grad_norm_move: list[int] of length 5
  first_loss_move: list[int] of length 5
  elapsed_s: float

part5:
  n_params: int
  flops_per_step: int                                        ← 6 × N_params × B × T
  avg_step_ms: float
  achieved_tflops: float
  peak_fp32_tflops: float                                    ← 13.0 (RTX 3060 spec)
  peak_bf16_tflops: float                                    ← 26.0 (RTX 3060 spec)
  mfu_fp32_pct: float
  mfu_bf16_pct: float

part6:
  fp32:    {bits, sign, exponent, mantissa, bias_exp=127, bias_mant=23, decoded, true, format}
  bf16:    {bits, sign, exponent, mantissa, bias_exp=127, bias_mant=7,  decoded, true, format}
  fp8_e4m3:{bits, sign, exponent, mantissa, bias_exp=7,   bias_mant=3,  decoded, true, format,
            subnormal: bool}

part7_summary: {device, torch, cuda, platform, n_params, peak_fp32_tflops, peak_bf16_tflops,
                achieved_tflops, mfu_fp32_pct, mfu_bf16_pct, avg_step_ms, tokens_per_sec,
                initial_loss, final_loss, loss_reduction_pct, total_steps}

device, torch, cuda, platform: str                          ← shown in header
```

### 6.2 `artifacts/metrics.json`

Top-level keys: `summary` (same shape as `bundle.part7_summary`), `metrics`.

```
metrics:
  steps:            list[int]   of length N     ← 0 .. N-1
  losses:           list[float] of length N     ← per-step cross-entropy
  grad_norms:       list[float] of length N     ← per-step ‖∇‖ after clipping
  step_times_ms:     list[float] of length N     ← per-step wall time (ms)
  tokens_per_step:  list[int]   of length N     ← B × T (here 1024)
  lr:    float                                  ← optimiser LR (3e-3)
  batch: int                                    ← micro-batch size (16)
  seq_len: int                                  ← sequence length (64)
```

The three arrays are equal length and index-aligned: `losses[i]` is the loss at step `i`,
etc. The dashboard reads them straight into the loss/grad-norm and step-time charts.

### 6.3 `artifacts/final_p[1-7].png`

Reference screenshots from the last verification pass. They are *not* regenerated by
`train.py` — they live in the repo as a record of what the dashboard looked like at the
time of writing.

| file | what it shows |
|---|---|
| `final_p1.png` | Shape tour — full tensor table + parameter / config / naming-key cards |
| `final_p2.png` | Gradient check — three result cards (point_0_0, point_1_2, tensor_mean) + math + "what can go wrong" |
| `final_p3.png` | Broken accumulation — both curves plotted + the formula comparison |
| `final_p4.png` | Loss vs grad-norm with the leading step marked |
| `final_p5.png` | MFU bar chart with 40 % reference line + raw numbers table |
| `final_p6.png` | 0.1 in fp32, bf16, fp8 E4M3 with bit visualisations |
| `final_p7.png` | Training-loop charts (loss + grad-norm combined, per-step time, zoomed tail) + summary table |

---

## 7. The dashboard (`index.html`)

Self-contained: no CDN, no server, no fetch. The training JSON is baked into the file by
`train.py` (see §9) so opening the file from disk just works.

### 7.1 Structure

```
┌────────────────────────────────────────────────────────┐
│  HERO — title, subtitle, status badge, env metadata    │
├────────────────────────────────────────────────────────┤
│  KPI ROW — 8 cards: params, loss, tok/s, step, MFU×2,  │
│              achieved TFLOPS, steps                     │
├────────────────────────────────────────────────────────┤
│  TABS — 1. Shape tour | 2. Grad check | 3. Accum |      │
│         4. Grad-norm | 5. MFU | 6. FP formats | 7. Loop │
├────────────────────────────────────────────────────────┤
│  ACTIVE TAB content                                     │
│   p1: shapes table + parameter / config / naming cards  │
│   p2: 3 grad-check cards + math + bugs list            │
│   p3: formula cards + chart                             │
│   p4: chart with lead-step marker + first-move lists   │
│   p5: numbers table + bar chart w/ 40% ref line        │
│   p6: 3 format cards with colour-coded bit rows        │
│   p7: 3 charts (loss+gn / step-time / zoomed tail)      │
│       + summary table                                   │
├────────────────────────────────────────────────────────┤
│  FOOTER — timestamp                                     │
└────────────────────────────────────────────────────────┘
```

### 7.2 Charts

All charts are vanilla `<canvas>` rendered by a small helper module baked into the HTML
(no Chart.js, no D3). Each chart:

- High-DPI aware (`devicePixelRatio` scaling)
- Tick labels use nice numbers (`niceTicks` rounds to 1/2/5 × 10ⁿ)
- Re-paints when the tab becomes visible (handles the 0×0-canvas-on-hidden-tab trap)

### 7.3 Bit visualisations (part 6)

Each format is rendered as a row of three labelled groups: `sign`, `exponent`, `mantissa`.
Each bit is a `<span class="bit">` — a coloured square with `0` or `1` inside. Colour key:

- magenta = sign bit
- green = exponent bits
- cyan = mantissa bits
- grey = 0 bit (dark cell)
- bright = 1 bit (filled cell)

Below the bit row: the decoded decimal, the true decimal, the exponent bias, and (for fp8)
whether the value is subnormal.

### 7.4 What the dashboard does *not* do

- No interactivity beyond tab switching (this is a *report*, not a controls panel)
- No live updates — refreshing the page after re-running `train.py` is the update path
- No data downloads — `artifacts/*.json` are the canonical source

---

## 8. The notebook (`notebook.ipynb`)

Same logic as `train.py`, organised into 26 cells with extensive markdown between them:

```
cell  1   imports + env probe
cell  2   mkdir artifacts/
cell  3   PART 1 — instantiate model, print config
cell  4   PART 1 — print_shapes tour (the 34-tensor table)
cell  5   PART 1 — how to read this
cell  6   PART 2 — pick a parameter, run grad check
cell  7   PART 2 — interpret results
cell  8   PART 3 — broken-accumulation simulation
cell  9   PART 3 — plot
cell 10   PART 3 — interpretation
cell 11   PART 4 — real-loop grad-norm + loss logger
cell 12   PART 4 — find lead step, plot
cell 13   PART 4 — interpretation
cell 14   PART 5 — MFU timing
cell 15   PART 5 — interpretation
cell 16   PART 6 — fp32 / bf16 / fp8 bit decomposition
cell 17   PART 6 — verdict
cell 18   PART 7 — define run_real_loop
cell 19   PART 7 — call it
cell 20   PART 7 — read the dashboard
```

Run it:

```bash
conda activate pytorch
jupyter nbconvert --to notebook --execute notebook.ipynb \
    --output notebook_executed.ipynb --ExecutePreprocessor.timeout=600
# or open it in jupyter lab
```

The pre-executed `notebook_executed.ipynb` ships in the repo as a record of what the cells
produce when run on the 3060.

---

## 9. How the bake-in works

`train.py` does not load `index.html` over HTTP. After producing `bundle.json` and
`metrics.json` it rewrites `index.html` directly on disk:

```
artifacts/bundle.json    ──┐
                            │  bake_index_html()
artifacts/metrics.json   ──┴── regex match: const BUNDLE  = /*__BUNDLE__*/  <JSON> ;
                                  regex match: const METRICS = /*__METRICS__*/ <JSON> ;
                                  replace <JSON> with the new JSON, write file back
```

The placeholder `/*__BUNDLE__*/` and `/*__METRICS__*/` are valid JS comments so the file
parses even when empty. The regex is greedy but anchored to `;`, so it works whether the
JSON after the placeholder is `{}` (first run, fresh repo) or a previous bake's payload
(re-run).

To force a fresh dashboard before a re-bake:

```bash
python train.py --steps 400 --reset-placeholders
```

This rewrites both `const BUNDLE = /*__BUNDLE__*/ {};` and `const METRICS = /*__METRICS__*/ {};`
back to the empty form, *then* runs the bake. Useful after editing the dashboard.

---

## 10. Reproducing on different hardware

The numbers in the bundle are pinned to the 3060 Laptop. On a different GPU they will
change. Most of the code is hardware-agnostic — the only hardware-dependent numbers are:

- `peak_fp32_tflops` and `peak_bf16_tflops` (theoretical peak from the GPU's spec sheet)
- The achieved TFLOPS and the resulting MFU percentages
- The `device`, `torch`, `cuda`, `platform` fields

To run on a different machine, just `python train.py` — the dashboard will pick up the
new numbers automatically. If your GPU is *much* faster than the 3060, you may want more
steps (`--steps 2000`) so the loss curve is more interesting.

---

## 11. What we did *not* do, and why

These were deliberately left out so the assignment's six truths stay the focus:

- **No bf16 training path.** The loop runs in fp32 throughout. We discussed in part 5 why
  that costs us MFU; we did not change it because doing so would change the autograd tape
  and complicate parts 2, 4, and 7 in non-obvious ways. Adding `--dtype bf16` would be a
  sensible follow-up.
- **No `torch.compile`.** Same reason — keep the measurements clean.
- **No real dataset.** The synthetic arithmetic-with-noise task is enough to demonstrate
  that the loop *works* (loss goes down, gradients flow) without pulling in a 50-GB download.
- **No wandb / tensorboard.** The dashboard already serves the same role, and we wanted it
  offline-capable.
- **No checkpointing / saving the model.** Single run, single purpose.
- **No distributed training.** One GPU, one process.

---

## 12. Honest lessons

The point of this assignment was not to produce a model that performs well — the model
is *deliberately* too small to do anything useful. The point was to internalise six
debugging habits:

1. **Print the shapes.** Every serious training bug that survives the first compile is
   either a shape error (caught by prints) or a silent loss error (caught by everything
   else on this list). Print shapes first.
2. **Check one gradient.** `loss.backward()` is a black box. Forgetting `zero_grad()`,
   forgetting `backward()`, accidentally wrapping something in `no_grad()`, accidentally
   casting a buffer to int — each of these silently produces nonsense gradients. The
   central-difference test costs ~10 ms and catches them all.
3. **Be aware of your accumulation math.** When micro-batches have different sizes,
   `mean(losses)` lies. The fix is one line.
4. **Watch the grad norm.** The loss curve lags. The grad norm is the early-warning
   system that lets you clip / lower LR / stop before the loss actually explodes.
5. **Report MFU honestly.** If you don't know what fraction of peak you're hitting, you
   don't know whether to spend your time on the model or on the harness. Measure it.
6. **Respect the format.** Bf16 is the right training precision for most of 2026-era
   hardware. fp8 is a throughput optimisation, not a precision upgrade. fp32 is what
   your optimiser state lives in.

The dashboard is meant to make all six of those checks obvious in one view, with the
actual numbers from a real run on real hardware.