"""
Train Mini-GPT on input.txt (Tiny Shakespeare) and save the weights to mini_gpt.pt.

Steps:
    1. Read input.txt and tokenize it with the GPT-2 BPE tokenizer (tiktoken)
    2. Split tokens 90% train / 10% validation
    3. Train with AdamW on random (input, next-token target) batches
    4. Print train/val loss every eval_interval steps
    5. Generate a short sample to sanity-check the model
    6. Save the trained weights to mini_gpt.pt (used by app.py)

Run:
    python train.py
"""

import torch
import torch.nn.functional as F
import tiktoken # 1. Import tiktoken
from model import MiniGPT

# -----------------
# 1. Dataset & Tokenizer Setup (Upgraded to BPE)
# -----------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Load the GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab # This will be 50,257

# Redefine encode and decode wrappers to match our previous API
encode = lambda s: enc.encode(s, allowed_special="all")   # str -> list[int]
decode = lambda l: enc.decode(l)                          # list[int] -> str

# Encode the text into a tensor
print("Encoding dataset with BPE... this might take a few seconds.")
data = torch.tensor(encode(text), dtype=torch.long)
# First 90% of tokens for training, last 10% for validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)  # for reproducible batches / initialization

# -----------------
# 2. Hyperparameters
# -----------------
batch_size = 32       # sequences per training step
block_size = 64       # context length (tokens per sequence)
max_iters = 1000      # total training steps
eval_interval = 100   # how often to print train/val loss
learning_rate = 1e-3

# Model architecture params
# NOTE: app.py hard-codes the same values; if you change them here, change them there too.
n_embd = 64
n_head = 4
n_layer = 4
# Pick the fastest available device: NVIDIA GPU -> Apple Silicon GPU -> CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# -----------------
# 3. Batcher
# -----------------
def get_batch(split):
    """Sample batch_size random windows of block_size tokens.

    x is the input window, y is the same window shifted one token to the right,
    so y[t] is the "next token" the model should predict after seeing x[:t+1].
    """
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))   # random start offsets
    x = torch.stack([data_split[i : i+block_size] for i in ix])
    y = torch.stack([data_split[i+1 : i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# -----------------
# 4. Evaluation Helper
# -----------------
@torch.no_grad() # Tell PyTorch not to build computation graphs for evaluation
def estimate_loss(model):
    """Average the loss over 50 random batches for both train and val splits (less noisy than one batch)."""
    out = {}
    model.eval() # Set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            logits, _ = model(X)
            # Reshape for Cross Entropy: (B, T, vocab_size) -> (B*T, vocab_size)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = Y.view(B * T)
            loss = F.cross_entropy(logits, targets)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Set back to training mode
    return out

# -----------------
# 5. Initialization & Training Loop
# -----------------
model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting training for {max_iters} iterations...")
for iter in range(max_iters):

    # Evaluate the loss periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass: predict the next token at every position, compare with targets
    logits, _ = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = yb.view(B * T)
    loss = F.cross_entropy(logits, targets)

    # Backward pass: clear old gradients, compute new ones, update weights
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training complete!")

# -----------------
# 6. Generate Text!
# -----------------
print("\n--- Generation Test ---")
# Start from token id 0. (With the GPT-2 BPE tokenizer, id 0 is the "!" character.)
context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("Generating 200 characters with KV Cache...")
# Try tweaking temperature and top_k!
# Note: max_new_tokens=200 is more than block_size (64). On CPU this raises an
# IndexError in the position embedding (see "Known limitations" in README.md).
generated_indices = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=10)

# Decode the generated tensor back to string
generated_text = decode(generated_indices[0].tolist())
print(generated_text)

# -----------------
# 7. Save Weights
# -----------------
# Only the state_dict (weights) is saved; app.py rebuilds the model with the same hyperparameters.
torch.save(model.state_dict(), 'mini_gpt.pt')
