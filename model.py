"""
Mini-GPT model definition.

A small decoder-only Transformer (GPT-style) written from scratch in PyTorch.

Architecture overview (one forward pass):
    token ids (B, T)
      -> token embedding + position embedding       (B, T, n_embd)
      -> n_layer x Block:
             x = x + CausalSelfAttention(LayerNorm(x))
             x = x + MLP(LayerNorm(x))
      -> final LayerNorm
      -> lm_head (Linear to vocab_size)              (B, T, vocab_size) logits

Shape legend used in the comments below:
    B  = batch size
    T  = sequence length (number of tokens in this forward pass)
    C  = n_embd (embedding / channel dimension)
    nh = n_head (number of attention heads)
    hs = head size = C // nh

The model also supports a KV cache: during generation, the keys and values of
past tokens are stored so each new step only has to process one new token
instead of re-running the whole sequence.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention: each token can only attend to itself and earlier tokens."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"

        self.n_head = n_head
        self.n_embd = n_embd

        # Key, Query, Value projections bundled into a single Linear layer for efficiency
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Causal mask: Lower triangular matrix of 1s.
        # Shape (1, 1, block_size, block_size) so it broadcasts over batch and heads.
        # Stored as a buffer: it moves with .to(device) and is saved in the state_dict,
        # but it is not a trainable parameter.
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x, use_cache=False, layer_past=None):
        # x: (B, T, C)
        # use_cache:  if True, return this layer's (k, v) so the caller can reuse them next step
        # layer_past: (past_k, past_v) from previous steps, each (B, nh, T_past, hs), or None
        B, T, C = x.size()

        # 1. Project to Q, K, V
        qkv = self.c_attn(x)                        # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)     # each (B, T, C)

        # 2. Split channels into heads: (B, T, C) -> (B, nh, T, hs)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # --- KV CACHE LOGIC ---
        # Prepend the cached keys/values from earlier tokens, so the new query
        # can attend to the whole history without recomputing it.
        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat([past_k, k], dim=-2) # Concat along time dimension
            v = torch.cat([past_v, v], dim=-2)

        present = (k, v) if use_cache else None
        # ----------------------

        # 3. Compute Attention scores, scaled by 1/sqrt(hs) to keep softmax well-behaved
        # (B, nh, T, hs) @ (B, nh, hs, T_total) -> (B, nh, T, T_total)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (hs ** 0.5))

        # 4. Causal masking: block attention to future positions.
        # When T == 1 (a single new token using the KV cache) there is no future
        # to hide, so the mask is skipped.
        # Note: this assumes that when T > 1 there is no cache (i.e. T_total == T).
        if T > 1:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)

        # 5. Aggregate Values: weighted sum of values for each query
        y = att @ v                                          # (B, nh, T, hs)
        # Merge heads back: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y, present

class Block(nn.Module):
    """One Transformer block (pre-LayerNorm): attention + feed-forward MLP, each with a residual connection."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)

        self.ln_2 = nn.LayerNorm(n_embd)
        # Position-wise feed-forward network: expand 4x, GELU, project back
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, use_cache=False, layer_past=None):
        # Tokens communicate with each other (attention) ...
        attn_out, present = self.attn(self.ln_1(x), use_cache=use_cache, layer_past=layer_past)
        x = x + attn_out
        # ... then each token is processed independently (MLP)
        x = x + self.mlp(self.ln_2(x))
        return x, present

class MiniGPT(nn.Module):
    """
    The full language model.

    Args:
        vocab_size: number of tokens in the tokenizer (50,257 for GPT-2 BPE)
        n_embd:     embedding dimension
        n_head:     number of attention heads per block
        n_layer:    number of Transformer blocks
        block_size: maximum context length (number of positions the model knows about)
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size

        # Learned lookup tables: token id -> vector, position index -> vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        # Projects each final hidden vector to a score (logit) for every vocab token
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, use_cache=False, past_key_values=None):
        """
        Args:
            idx:             LongTensor of token ids, shape (B, T)
            use_cache:       if True, also return the per-layer (k, v) cache
            past_key_values: list with one (k, v) tuple per layer from a previous call, or None

        Returns:
            logits:   (B, T, vocab_size) scores for the next token at every position
            presents: list of per-layer (k, v) tuples if use_cache else None
        """
        B, T = idx.size()

        # Number of tokens already in the cache. New tokens get positions after them.
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0

        tok_emb = self.token_embedding_table(idx)                                          # (B, T, C)
        pos = torch.arange(past_length, past_length + T, dtype=torch.long, device=idx.device)  # (T,)
        # Note: position_embedding_table only has block_size rows, so
        # past_length + T must stay <= block_size or this lookup fails.
        pos_emb = self.position_embedding_table(pos)                                       # (T, C)

        x = tok_emb + pos_emb   # (B, T, C), position embedding broadcasts over batch

        presents = [] if use_cache else None

        # Run through every Transformer block, feeding each its own slice of the cache
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, use_cache=use_cache, layer_past=layer_past)
            if use_cache:
                presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        return logits, presents
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Takes a conditioning sequence of indices idx (LongTensor of shape (B, T)) and completes
        the sequence max_new_tokens times, feeding the predictions back into the model each time.

        Decoding knobs:
            temperature: divides the logits; < 1.0 = more confident / repetitive, > 1.0 = more random
            top_k:       if set, only sample from the k most likely tokens
            top_p:       if set, only sample from the smallest set of tokens whose
                         cumulative probability exceeds top_p (nucleus sampling)

        Steps per new token:
            1. Run the model (full prompt on the first step, then only the newest token + KV cache)
            2. Take the logits for the last position and apply temperature
            3. Optionally filter with top-k and/or top-p
            4. Sample one token and append it to the sequence
        """
        # Store KV cache across generation steps
        past_key_values = None

        for _ in range(max_new_tokens):
            # If we have a cache, we only need to pass the last token
            if past_key_values is not None:
                idx_cond = idx[:, -1:]
            else:
                # If sequence is longer than block_size, crop context to the last block_size tokens
                idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Forward pass
            logits, past_key_values = self(idx_cond, use_cache=True, past_key_values=past_key_values)

            # Pluck the logits at the final step and scale by desired temperature
            # logits shape is (B, T, vocab_size). We want (B, vocab_size)
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits that are smaller than the kth largest to -inf
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Optionally apply Top-P (nucleus) sampling
            if top_p is not None:
                # Sort probabilities descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold (top_p)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter the removal mask back to the original indices
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                # Renormalize
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)   # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# --- TEST SCRIPT ---
# Run `python model.py` to check that the KV cache is correct:
# the logits for token 5 must be the same whether we run all 5 tokens at once,
# or run 4 tokens first and then only the 5th token using the cache.
if __name__ == '__main__':
    model = MiniGPT(vocab_size=65, n_embd=32, n_head=4, n_layer=2, block_size=16)

    # 1. Standard Forward Pass (like training)
    idx_seq = torch.randint(0, 65, (1, 5)) # 5 tokens
    logits_no_cache, _ = model(idx_seq)

    # 2. KV-Cache Forward Pass (like generation)
    idx_past = idx_seq[:, :4]
    _, past_key_values = model(idx_past, use_cache=True)

    idx_new = idx_seq[:, 4:] # Just the 5th token
    logits_with_cache, _ = model(idx_new, use_cache=True, past_key_values=past_key_values)

    # 3. Compare the prediction for the last token from both paths
    diff = (logits_no_cache[:, -1, :] - logits_with_cache[:, 0, :]).abs().max().item()
    print(f"Max difference between standard and cached logits: {diff:.6f}")
    assert diff < 1e-5, "KV Cache implementation is incorrect!"
    print("Success: KV-Cache produces mathematically identical results for the next token!")
