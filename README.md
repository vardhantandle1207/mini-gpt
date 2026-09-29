# Mini-GPT

A small GPT-style (decoder-only Transformer) language model written from scratch in PyTorch, trained on the Tiny Shakespeare dataset, and served through a Gradio web app.

**Features**

- Multi-head causal self-attention, pre-LayerNorm Transformer blocks, GELU MLP
- GPT-2 BPE tokenizer (via `tiktoken`, vocab size 50,257)
- **KV cache** for faster generation: after the first step, only the newest token goes through the model
- Decoding with **temperature**, **top-k**, and **top-p (nucleus)** sampling
- A built-in self-test that checks the KV cache gives the same results as a normal forward pass
- Gradio UI, ready to deploy as a Hugging Face Space

**Results** (checkpoint in this repo, averaged over 200 random batches of 32×64 tokens)

| Split | Loss | Perplexity |
|---|---|---|
| Train | 4.31 | 75 |
| Val | 4.90 | 134 |

For reference, a random model starts at ln(50,257) ≈ 10.8. The gap between train and val shows the model is starting to overfit, which is expected with no dropout on ~300k training tokens.

---

## Project structure

| File | What it does |
|---|---|
| [model.py](model.py) | Model definition: `CausalSelfAttention`, `Block`, `MiniGPT` (forward pass + `generate()`). Run it directly to test the KV cache. |
| [train.py](train.py) | Tokenizes `input.txt`, trains the model, prints losses, generates a sample, and saves `mini_gpt.pt`. |
| [app.py](app.py) | Loads `mini_gpt.pt` and starts the Gradio web interface. |
| [bench.py](bench.py) | Measures generation speed (tokens/sec) with and without the KV cache. |
| `input.txt` | Training text (Tiny Shakespeare, ~1.1 MB, ~338k BPE tokens). |
| `mini_gpt.pt` | Trained weights plus the model hyperparameters (~26 MB). Created by `train.py`. |
| [requirements.txt](requirements.txt) | Python dependencies: `torch`, `tiktoken`, `gradio`. |

---

## Model

```
token ids (B, T)
  → token embedding + position embedding        (B, T, 64)
  → 4 × Transformer Block
        x = x + CausalSelfAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))               # 64 → 256 → 64, GELU
  → final LayerNorm
  → lm_head (Linear 64 → 50,257)                (B, T, 50257) logits
```

| Hyperparameter | Value |
|---|---|
| Embedding size (`n_embd`) | 64 |
| Attention heads (`n_head`) | 4 (head size 16) |
| Layers (`n_layer`) | 4 |
| Context length (`block_size`) | 64 tokens |
| Vocabulary | 50,257 (GPT-2 BPE) |
| Parameters | ~6.6M (most of them are in the token embedding and `lm_head`) |

### Training setup

| Setting | Value |
|---|---|
| Train / val split | 90% / 10% |
| Batch size | 32 |
| Iterations | 1,000 |
| Optimizer | AdamW, learning rate 1e-3 |
| Loss | Cross-entropy on next-token prediction |
| Device | CUDA → Apple MPS → CPU (picked automatically) |

---

## Getting started

### 1. Install

```bash
git clone <your-repo-url>
cd mini-gpt

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. (Optional) Check the model and KV cache

```bash
python model.py
```

Expected output:

```
Max difference between standard and cached logits: 0.000000
Success: KV-Cache produces mathematically identical results for the next token!
```

### 3. Train

```bash
python train.py
```

This will:

1. Read `input.txt` and encode it with the GPT-2 tokenizer
2. Train for 1,000 steps, printing train/val loss every 100 steps
3. Print a generated sample
4. Save the weights and hyperparameters to `mini_gpt.pt`

A trained `mini_gpt.pt` is already included, so you can skip this step and go straight to the app.

### 4. Run the web app

```bash
python app.py
```

Open the URL Gradio prints (usually http://127.0.0.1:7860), enter a prompt such as `ROMEO:`, and click **Generate Text**.

The app and `bench.py` need `mini_gpt.pt`. They read the model's hyperparameters from it, so they keep working if you change the architecture in `train.py` and retrain.

---

## Decoding parameters

| Parameter | Effect |
|---|---|
| **Max New Tokens** | How many tokens to generate after the prompt. |
| **Temperature** | Divides the logits before softmax. Lower (e.g. 0.5) = safer and more repetitive; higher (e.g. 1.5) = more random. |
| **Top-K** | Only sample from the K most likely next tokens. |
| **Top-P (Nucleus)** | Only sample from the smallest set of tokens whose combined probability is ≥ P. |

Top-k is applied first, then top-p, then one token is sampled.

---

## How the KV cache works

Without a cache, generating each new token would re-run attention over the whole sequence.
With the cache:

1. **First step:** the full prompt is passed through the model. Each attention layer returns its keys and values `(k, v)`.
2. **Every later step:** only the newest token is passed in. Each layer computes `k, v` for that one token, concatenates them with the cached ones, and attends over the full history.
3. Position embeddings continue from where the cache left off (`past_length`).

`python model.py` checks this: the logits for the 5th token must be the same whether you run all 5 tokens at once, or run 4 tokens and then the 5th token with the cache.

### Benchmark

```bash
python bench.py
```

This generates 63 tokens from a 1-token prompt (the most the 64-token context allows) with greedy decoding, once re-running the full sequence at every step and once using the KV cache. It times 20 runs of each and checks that both methods produce the same tokens. Example results on an Apple Silicon Mac:

| Device | No cache | KV cache | Speedup |
|---|---|---|---|
| CPU | ~680 tok/s | ~2,260 tok/s | **~3.3×** |
| MPS (Apple GPU) | ~157 tok/s | ~153 tok/s | ~1.0× |

On CPU the cache gives a clear speedup. On MPS it doesn't help: the model is so small that per-step GPU launch overhead dominates, so skipping work saves almost nothing. The speedup is also limited by the short 64-token context, because longer sequences have more repeated work for the cache to skip. Numbers vary between runs and machines.

---

## Deploying to Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK.
2. In the Space's copy of `README.md`, put this configuration block at the very top (it is left out of this GitHub README so it doesn't show up as a table):

   ```yaml
   ---
   title: Mini-GPT
   app_file: app.py
   sdk: gradio
   sdk_version: 6.28.0
   ---
   ```

3. Push `app.py`, `model.py`, `requirements.txt`, `README.md`, and `mini_gpt.pt` to it.
4. The Space installs `requirements.txt` and runs `app.py`.

---

## Known limitations

- **Generation length is capped by `block_size` (64 tokens).** The position embedding table only has 64 rows, and with the KV cache the position keeps growing past that. Once **prompt tokens + new tokens > 64**, generation fails with `IndexError: index out of range in self` on CPU. On GPU/MPS it can fail differently or give bad output. This affects:
  - the app's default setting of 150 new tokens (and the slider, which goes up to 500)
  - the 200-token sample at the end of `train.py`

  Workaround for now: keep `Max New Tokens` + prompt length under 64. A proper fix is to drop the cache and re-crop the context to the last `block_size` tokens once the sequence gets longer than the window.
- **Small model, short training run.** Expect Shakespeare-*flavoured* text, not coherent sentences.
- An empty prompt starts generation from token id 0, which is the `!` character in the GPT-2 tokenizer.

---

## Ideas for improvement

- Fix the context-window limitation above (sliding window + cache reset)
- Add dropout and a learning-rate schedule
- Tie the weights of `token_embedding_table` and `lm_head` (cuts ~3.2M parameters)
- Stream tokens to the Gradio UI as they are generated
