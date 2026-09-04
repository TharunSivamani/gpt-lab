#!/usr/bin/env python
"""train.py — the assignment in production form.

Runs all six investigations, then a real training loop, and dumps
everything to artifacts/metrics.json for the dashboard to consume.

Usage:
    conda activate pytorch
    python train.py            # ~90 s on a 3060
    python train.py --steps 800   # longer
"""
from __future__ import annotations
import argparse, json, math, os, platform, re, struct, sys, time
from pathlib import Path

import torch

# local
sys.path.insert(0, str(Path(__file__).parent))
from tiny_gpt import (
    TinyGPT, GPTConfig,
    print_shapes, manual_grad_check_scalar, broken_grad_accum_loss,
    PEAK_FP32_TFLOPS, PEAK_BF16_TFLOPS,
    make_synthetic_batch, cuda_sync_timer,
    _qualified_name,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ART = Path('artifacts'); ART.mkdir(exist_ok=True)


def bake_index_html(html_path: Path, bundle: dict):
    """Replace the data behind the placeholder markers with this run's JSON.

    Looks for lines of the form:
        const BUNDLE  = /*__BUNDLE__*/  <any JSON value> ;
        const METRICS = /*__METRICS__*/ <any JSON value> ;

    and rewrites them. Idempotent — works whether the placeholders are still
    empty ({}) or have already been baked once before.
    """
    if not html_path.exists():
        print(f'(no {html_path} found — skipping HTML bake)')
        return
    metrics = json.loads((ART / 'metrics.json').read_text())
    html = html_path.read_text()
    bundle_json = json.dumps(bundle).replace('</', '<\\/')
    metrics_json = json.dumps(metrics).replace('</', '<\\/')
    pattern_bundle = re.compile(r'(const\s+BUNDLE\s*=\s*/\*__BUNDLE__\*/\s*)(.*?)(\s*;)', re.DOTALL)
    pattern_metrics = re.compile(r'(const\s+METRICS\s*=\s*/\*__METRICS__\*/\s*)(.*?)(\s*;)', re.DOTALL)
    n = 0
    new = pattern_bundle.sub(lambda m: m.group(1) + bundle_json + m.group(3), html)
    if new != html:
        html = new; n += 1
    new = pattern_metrics.sub(lambda m: m.group(1) + metrics_json + m.group(3), html)
    if new != html:
        html = new; n += 1
    if n:
        html_path.write_text(html)
        print(f'baked {n} JSON block(s) into {html_path}')
    else:
        print(f'(no bundle/metrics constants found in {html_path} — leaving alone)')


# ---------------------------------------------------------------------------
# Part 1: shape tour
# ---------------------------------------------------------------------------
def part1_shape_tour():
    cfg = GPTConfig(vocab_size=512, block_size=32, n_layer=3, n_head=4, n_embd=128)
    model = TinyGPT(cfg).to(DEVICE)
    info = {
        'cfg': {k: getattr(cfg, k) for k in ('vocab_size','block_size','n_layer','n_head','n_embd')},
        'params_total': model.num_params(),
        'params_non_embedding': model.num_params(exclude_embedding=True),
        'weight_tying': True,
        'shapes': [],
    }
    # capture shapes
    cfg_ = cfg; B, T = 2, 16
    x = torch.randint(0, cfg_.vocab_size, (B, T), device=DEVICE)
    info['shapes'].append({'name': 'idx (input tokens)', 'shape': list(x.shape), 'meaning': '(B,T) — token ids'})
    tok = model.tok_emb(x); info['shapes'].append({'name':'tok_emb','shape':list(tok.shape),'meaning':'(B,T,C) — token embeddings'})
    pos = torch.arange(0, T, device=DEVICE); info['shapes'].append({'name':'pos_idx','shape':list(pos.shape),'meaning':'(T,) — positions 0..T-1'})
    pe = model.pos_emb(pos); info['shapes'].append({'name':'pos_emb','shape':list(pe.shape),'meaning':'(T,C) — positional embeddings'})
    x_in = tok + pe
    info['shapes'].append({'name':'residual stream (tok+pos)','shape':list(x_in.shape),'meaning':'(B,T,C) — residual stream entry'})
    for i, blk in enumerate(model.blocks):
        h = blk.ln_1(x_in); info['shapes'].append({'name':f'block{i}.ln_1','shape':list(h.shape),'meaning':'(B,T,C) — pre-attn norm'})
        qkv = blk.attn.c_attn(h); info['shapes'].append({'name':f'block{i}.c_attn (QKV)','shape':list(qkv.shape),'meaning':'(B,T,3C) — fused Q,K,V projection'})
        q,k,v = qkv.split(cfg_.n_embd, dim=2); info['shapes'].append({'name':f'block{i}.Q/K/V','shape':list(q.shape),'meaning':'(B,T,C) — per-head before reshape'})
        nh = cfg_.n_head; hd = cfg_.n_embd // nh
        q = q.view(B,T,nh,hd).transpose(1,2)
        k = k.view(B,T,nh,hd).transpose(1,2)
        v = v.view(B,T,nh,hd).transpose(1,2)
        info['shapes'].append({'name':f'block{i}.Q reshaped','shape':list(q.shape),'meaning':'(B,n_head,T,head_dim) — heads as separate batch dim'})
        ctx = torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        info['shapes'].append({'name':f'block{i}.attn out','shape':list(ctx.shape),'meaning':'(B,n_head,T,head_dim) — attended values'})
        y = ctx.transpose(1,2).contiguous().view(B,T,cfg_.n_embd)
        proj = blk.attn.c_proj(y); info['shapes'].append({'name':f'block{i}.c_proj','shape':list(proj.shape),'meaning':'(B,T,C) — output projection'})
        x_in = x_in + proj
        info['shapes'].append({'name':f'block{i}.residual + attn','shape':list(x_in.shape),'meaning':'(B,T,C) — residual stream after attention'})
        m = blk.mlp(blk.ln_2(x_in)); info['shapes'].append({'name':f'block{i}.mlp','shape':list(m.shape),'meaning':'(B,T,C) — 4C expansion + projection back'})
        x_in = x_in + m
        info['shapes'].append({'name':f'block{i}.residual + mlp','shape':list(x_in.shape),'meaning':'(B,T,C) — residual stream after MLP'})
    info['shapes'].append({'name':'ln_f','shape':list(model.ln_f(x_in).shape),'meaning':'(B,T,C) — final norm'})
    logits = model.head(model.ln_f(x_in))
    info['shapes'].append({'name':'logits','shape':list(logits.shape),'meaning':'(B,T,V) — vocab scores'})
    return info


# ---------------------------------------------------------------------------
# Part 2: gradient check
# ---------------------------------------------------------------------------
def part2_grad_check():
    cfg = GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=4, n_embd=64)
    torch.manual_seed(0)
    model = TinyGPT(cfg).to(DEVICE)
    target = None
    for n, p in model.named_parameters():
        if 'blocks.0.mlp.c_proj.weight' in n:
            target = p; break
    r = manual_grad_check_scalar(model, target, idx=(0, 0), eps=1e-2)
    r2 = manual_grad_check_scalar(model, target, idx=(1, 2), eps=1e-2)
    r3 = manual_grad_check_scalar(model, target, idx=None, eps=1e-2)  # tensor mean
    return {
        'point_0_0': r,
        'point_1_2': r2,
        'tensor_mean': r3,
        'all_agree_4dp': all(x.get('agree_4dp', False) for x in (r, r2, r3)),
        'param_qualified_name': _qualified_name(model, target),
    }


# ---------------------------------------------------------------------------
# Part 3: broken accumulation curve
# ---------------------------------------------------------------------------
def part3_broken_accum(STEPS=80, K=4, seed=1):
    """Show that 'average of averages' is wrong when micro-batches have
    unequal token counts. The gap is most visible early when the loss
    distribution is heavy-tailed across batches.
    """
    torch.manual_seed(seed)
    correct, broken = [], []
    base = 4.5
    for step in range(STEPS):
        # a step's true loss has a stable trend plus heavy-tailed per-batch noise
        true = base - 0.04 * step + 0.30 * (torch.rand(()).item() - 0.5)
        micro = []
        for _ in range(K):
            # token counts vary widely: 8..512
            n = int(torch.randint(8, 512, (1,)).item())
            # loss noise scales as 1/sqrt(n) — small batches are very noisy
            l = true + 1.0 / math.sqrt(n) * torch.randn(()).item()
            micro.append((l, n))
        c, b = broken_grad_accum_loss(micro)
        correct.append(c); broken.append(b)
    return {'correct': correct, 'broken': broken}


# ---------------------------------------------------------------------------
# Part 4: real loop logging grad norm + loss
# ---------------------------------------------------------------------------
def part4_real_loop_log(B=8, T=32, N=60):
    cfg = GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=4, n_embd=64)
    torch.manual_seed(42)
    m = TinyGPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    losses, grad_norms = [], []
    with cuda_sync_timer() as t:
        for step in range(N):
            x, y = make_synthetic_batch(B, T, cfg.vocab_size, DEVICE, seed=step)
            _, loss = m(x, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=float('inf')).item()
            opt.step()
            losses.append(loss.item()); grad_norms.append(gn)
    elapsed = t()
    # find first step where grad_norm leads loss by 2 steps
    def moves(series, thr):
        out = []
        prev = series[0]
        for i, v in enumerate(series):
            if abs(v - prev) > thr: out.append(i)
            prev = v
        return out
    nm = moves(grad_norms, 0.05); lm = moves(losses, 0.05)
    lead_step = None
    for s in nm:
        if not any(l == s + 2 for l in lm):
            lead_step = s; break
    return {
        'losses': losses, 'grad_norms': grad_norms,
        'lead_step': lead_step,
        'first_grad_norm_move': nm[:5],
        'first_loss_move': lm[:5],
        'elapsed_s': elapsed,
    }


# ---------------------------------------------------------------------------
# Part 5: MFU
# ---------------------------------------------------------------------------
def part5_mfu(B=16, T=64, N=30):
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=4, n_embd=128)
    torch.manual_seed(0)
    m = TinyGPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in m.parameters())
    flops_per_step = 6 * n_params * B * T
    for _ in range(3):
        x, y = make_synthetic_batch(B, T, cfg.vocab_size, DEVICE, seed=0)
        _, l = m(x, y); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(N):
        x, y = make_synthetic_batch(B, T, cfg.vocab_size, DEVICE, seed=s)
        _, l = m(x, y); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / N
    tflops = flops_per_step / dt / 1e12
    return {
        'n_params': n_params,
        'flops_per_step': flops_per_step,
        'avg_step_ms': round(dt * 1000, 3),
        'achieved_tflops': round(tflops, 4),
        'peak_fp32_tflops': PEAK_FP32_TFLOPS,
        'peak_bf16_tflops': PEAK_BF16_TFLOPS,
        'mfu_fp32_pct': round(tflops / PEAK_FP32_TFLOPS * 100, 2),
        'mfu_bf16_pct': round(tflops / PEAK_BF16_TFLOPS * 100, 2),
    }


# ---------------------------------------------------------------------------
# Part 6: 0.1 in three formats
# ---------------------------------------------------------------------------
def part6_fp_formats():
    x = 0.1
    out = {}

    # fp32
    b32 = ''.join(b for b in format(struct.unpack('<I', struct.pack('<f', float(x)))[0], '032b'))
    sign, exp, mant = b32[0], b32[1:9], b32[9:]
    val32 = (-1)**int(sign,2) * 2**(int(exp,2)-127) * (1 + int(mant,2)/2**23)
    out['fp32'] = {
        'bits': b32, 'sign': sign, 'exponent': exp, 'mantissa': mant,
        'bias_exp': 127, 'bias_mant': 23, 'decoded': val32, 'true': x,
        'format': 'IEEE-754 single precision, 1-8-23',
    }

    # bf16
    t = torch.tensor(float(x), dtype=torch.bfloat16)
    raw = t.view(torch.uint16).item()
    bb = format(raw, '016b')
    sb, eb, mb = bb[0], bb[1:9], bb[9:]
    val_bf = (-1)**int(sb,2) * 2**(int(eb,2)-127) * (1 + int(mb,2)/2**7)
    out['bf16'] = {
        'bits': bb, 'sign': sb, 'exponent': eb, 'mantissa': mb,
        'bias_exp': 127, 'bias_mant': 7, 'decoded': val_bf, 'true': x,
        'format': 'bfloat16, 1-8-7 (brain-float)',
    }

    # fp8 e4m3
    if hasattr(torch, 'float8_e4m3fn'):
        t8 = torch.tensor(float(x), dtype=torch.float8_e4m3fn)
        raw8 = t8.view(torch.uint8).item()
        b8 = format(raw8, '08b')
        s8, e8, m8 = b8[0], b8[1:5], b8[5:]
        exp_v = int(e8, 2)
        if exp_v == 0:
            val8 = (-1)**int(s8,2) * 2**(1-7) * (int(m8,2)/2**3)
            sub = True
        else:
            val8 = (-1)**int(s8,2) * 2**(exp_v-7) * (1 + int(m8,2)/2**3)
            sub = False
        out['fp8_e4m3'] = {
            'bits': b8, 'sign': s8, 'exponent': e8, 'mantissa': m8,
            'bias_exp': 7, 'bias_mant': 3, 'decoded': val8, 'true': x,
            'format': 'FP8 E4M3 (Hopper/Ada), 1-4-3',
            'subnormal': sub,
        }
    else:
        out['fp8_e4m3'] = {'error': 'torch.float8_e4m3fn not available on this build'}
    return out


# ---------------------------------------------------------------------------
# Part 7: real training loop, full diagnostics, JSON output
# ---------------------------------------------------------------------------
def part7_train(steps=400, B=16, T=64, lr=3e-3, log_json='artifacts/metrics.json'):
    cfg = GPTConfig(vocab_size=1024, block_size=64, n_layer=3, n_head=4, n_embd=128)
    torch.manual_seed(0)
    model = TinyGPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=torch.cuda.is_available())
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = 6 * n_params * B * T

    metrics = {'steps': [], 'losses': [], 'grad_norms': [],
               'step_times_ms': [], 'tokens_per_step': [],
               'lr': lr, 'batch': B, 'seq_len': T}
    # warmup
    for _ in range(3):
        x, y = make_synthetic_batch(B, T, cfg.vocab_size, DEVICE, seed=0)
        _, l = model(x, y); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for step in range(steps):
        x, y = make_synthetic_batch(B, T, cfg.vocab_size, DEVICE, seed=step)
        torch.cuda.synchronize(); st = time.perf_counter()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        opt.step()
        torch.cuda.synchronize(); dt = (time.perf_counter() - st) * 1000
        metrics['steps'].append(step)
        metrics['losses'].append(loss.item())
        metrics['grad_norms'].append(gn)
        metrics['step_times_ms'].append(dt)
        metrics['tokens_per_step'].append(B * T)
    total = time.perf_counter() - t0
    avg_dt = total / steps
    tflops = flops_per_step / avg_dt / 1e12
    summary = {
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'platform': platform.platform(),
        'n_params': n_params,
        'peak_fp32_tflops': PEAK_FP32_TFLOPS,
        'peak_bf16_tflops': PEAK_BF16_TFLOPS,
        'achieved_tflops': round(tflops, 4),
        'mfu_fp32_pct': round(tflops / PEAK_FP32_TFLOPS * 100, 2),
        'mfu_bf16_pct': round(tflops / PEAK_BF16_TFLOPS * 100, 2),
        'avg_step_ms': round(avg_dt * 1000, 3),
        'tokens_per_sec': round((B*T)/avg_dt),
        'initial_loss': metrics['losses'][0],
        'final_loss': metrics['losses'][-1],
        'loss_reduction_pct': round((1 - metrics['losses'][-1]/metrics['losses'][0]) * 100, 1),
        'total_steps': steps,
    }
    with open(log_json, 'w') as f:
        json.dump({'summary': summary, 'metrics': metrics}, f)
    return summary, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--seq', type=int, default=64)
    p.add_argument('--reset-placeholders', action='store_true',
                   help='Strip baked JSON from index.html back to empty {} before baking (useful when editing)')
    args = p.parse_args()

    if args.reset_placeholders and Path('index.html').exists():
        h = Path('index.html').read_text()
        h = re.sub(r'(const\s+BUNDLE\s*=\s*/\*__BUNDLE__\*/\s*).*?(\s*;)', r'\1{}\2', h, flags=re.DOTALL)
        h = re.sub(r'(const\s+METRICS\s*=\s*/\*__METRICS__\*/\s*).*?(\s*;)', r'\1{}\2', h, flags=re.DOTALL)
        Path('index.html').write_text(h)
        print('reset placeholders in index.html')

    print('=' * 78)
    print('PART 1 — shape tour')
    print('=' * 78)
    s1 = part1_shape_tour()
    print(f"params: {s1['params_total']:,} total, {s1['params_non_embedding']:,} non-embedding")
    print(f"shapes logged: {len(s1['shapes'])}")

    print('\n' + '=' * 78)
    print('PART 2 — gradient check (autograd vs central difference)')
    print('=' * 78)
    s2 = part2_grad_check()
    for tag, r in s2.items():
        if isinstance(r, dict) and 'analytic' in r:
            print(f"  {tag:14s}: analytic={r['analytic']: .10f}  numeric={r['numeric']: .10f}  rel_err={r.get('rel_err', 0):.3e}  agree_4dp={r.get('agree_4dp', False)}")
    print(f"  ALL agree to 4 decimal places (rel): {s2['all_agree_4dp']}")

    print('\n' + '=' * 78)
    print('PART 3 — broken accumulation curves')
    print('=' * 78)
    s3 = part3_broken_accum(STEPS=80)
    print(f"  correct_mean = {sum(s3['correct'])/len(s3['correct']):.3f}")
    print(f"  broken_mean  = {sum(s3['broken'])/len(s3['broken']):.3f}")

    print('\n' + '=' * 78)
    print('PART 4 — grad-norm-leads-loss')
    print('=' * 78)
    s4 = part4_real_loop_log()
    print(f"  lead_step (grad norm moves first by ≥2 steps): {s4['lead_step']}")

    print('\n' + '=' * 78)
    print('PART 5 — MFU')
    print('=' * 78)
    s5 = part5_mfu()
    print(f"  achieved {s5['achieved_tflops']:.3f} TFLOPS")
    print(f"  MFU vs FP32 peak ({s5['peak_fp32_tflops']} TFLOPS): {s5['mfu_fp32_pct']}%")
    print(f"  MFU vs BF16 peak ({s5['peak_bf16_tflops']} TFLOPS): {s5['mfu_bf16_pct']}%")

    print('\n' + '=' * 78)
    print('PART 6 — 0.1 in three formats')
    print('=' * 78)
    s6 = part6_fp_formats()
    for name, d in s6.items():
        if 'error' in d:
            print(f"  {name}: {d['error']}"); continue
        print(f"  {name:8s}  {d['format']}")
        print(f"           bits     : {d['bits']}")
        print(f"           sign|exp|mantissa : {d['sign']}|{d['exponent']}|{d['mantissa']}")
        print(f"           decoded  : {d['decoded']:.20f}")

    print('\n' + '=' * 78)
    print(f'PART 7 — full training loop ({args.steps} steps)')
    print('=' * 78)
    summary, _ = part7_train(steps=args.steps, B=args.batch, T=args.seq)
    print(f"  device           : {summary['device']}")
    print(f"  params           : {summary['n_params']:,}")
    print(f"  achieved         : {summary['achieved_tflops']} TFLOPS")
    print(f"  MFU FP32 / BF16  : {summary['mfu_fp32_pct']}% / {summary['mfu_bf16_pct']}%")
    print(f"  avg step         : {summary['avg_step_ms']} ms")
    print(f"  tokens/sec       : {summary['tokens_per_sec']:,}")
    print(f"  loss {summary['initial_loss']:.4f} -> {summary['final_loss']:.4f} "
          f"({summary['loss_reduction_pct']}% reduction)")

    # write the dashboard bundle
    bundle = {
        'part1': s1,
        'part2': s2,
        'part3': s3,
        'part4': s4,
        'part5': s5,
        'part6': s6,
        'part7_summary': summary,
        'device': summary['device'],
        'torch': summary['torch'],
        'cuda': summary['cuda'],
        'platform': summary['platform'],
    }
    (ART / 'bundle.json').write_text(json.dumps(bundle))
    print(f"\nwrote {ART/'bundle.json'} + {ART/'metrics.json'} for index.html")

    # Optional: bake the numbers into index.html so the dashboard can be opened
    # directly without a server. Idempotent — only rewrites if the placeholder is
    # present; otherwise leaves index.html alone.
    if os.environ.get('BAKE_HTML', '1') != '0':
        bake_index_html(Path('index.html'), bundle)


if __name__ == '__main__':
    main()