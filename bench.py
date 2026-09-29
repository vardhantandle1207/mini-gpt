"""
Benchmark generation speed with and without the KV cache.

Loads mini_gpt.pt and generates the same number of tokens two ways:
    - no cache: every step re-runs the model over the whole sequence so far
    - KV cache: the first step runs the prompt, every later step runs only the newest token

Both paths use greedy decoding (argmax), so they must produce identical tokens;
the script checks this. Sampling is left out on purpose: it costs the same in
both paths and would only add noise to the timing.

Run:
    python bench.py
"""

import time
import torch
from model import load_checkpoint

# -----------------
# 1. Settings
# -----------------
# The checkpoint stores the model's hyperparameters; only block_size is needed here.
block_size = torch.load('mini_gpt.pt', map_location='cpu', weights_only=True)['config']['block_size']

# 1 prompt token + 63 new tokens = 64 = block_size, the longest run the
# position embedding table allows (see "Known limitations" in README.md).
max_new_tokens = block_size - 1
warmup_runs = 3       # untimed runs first, so one-off setup costs don't skew the timing
timed_runs = 20

# Always benchmark on CPU, plus the GPU if there is one
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')
elif torch.backends.mps.is_available():
    devices.append('mps')

# -----------------
# 2. The two generation loops
# -----------------
@torch.no_grad()
def generate_no_cache(model, idx):
    """Greedy decoding that re-runs the full sequence through the model at every step."""
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)   # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def generate_kv_cache(model, idx):
    """Greedy decoding that feeds only the newest token and reuses the per-layer (k, v) cache."""
    past_key_values = None
    idx_cond = idx
    for _ in range(max_new_tokens):
        logits, past_key_values = model(idx_cond, use_cache=True, past_key_values=past_key_values)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)   # (B, 1)
        idx = torch.cat((idx, idx_next), dim=1)
        idx_cond = idx_next
    return idx

# -----------------
# 3. Timing helper
# -----------------
def synchronize(device):
    """GPU work runs asynchronously; wait for it to finish so the timer measures real work."""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

def tokens_per_sec(generate_fn, model, device):
    """Return (tokens/sec, generated ids) for one generation function."""
    # Start from token id 0 ("!" in GPT-2 BPE), same as train.py and app.py
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    for _ in range(warmup_runs):
        generate_fn(model, context)

    synchronize(device)
    start = time.perf_counter()
    for _ in range(timed_runs):
        out = generate_fn(model, context)
    synchronize(device)
    elapsed = time.perf_counter() - start

    return max_new_tokens * timed_runs / elapsed, out

# -----------------
# 4. Run the benchmark
# -----------------
print(f"Generating {max_new_tokens} tokens from a 1-token prompt, {timed_runs} timed runs per method\n")
print(f"{'device':<8}{'no cache':>14}{'KV cache':>14}{'speedup':>10}   same output")

for device in devices:
    model = load_checkpoint('mini_gpt.pt', device)
    model.eval()

    no_cache_tps, no_cache_out = tokens_per_sec(generate_no_cache, model, device)
    cache_tps, cache_out = tokens_per_sec(generate_kv_cache, model, device)
    same = torch.equal(no_cache_out, cache_out)

    print(f"{device:<8}{no_cache_tps:>10.0f} t/s{cache_tps:>10.0f} t/s{cache_tps / no_cache_tps:>9.2f}x   {same}")
