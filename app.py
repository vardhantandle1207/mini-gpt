"""
Gradio web app for Mini-GPT.

Loads the trained weights from mini_gpt.pt and serves a simple UI where you type
a prompt, adjust the decoding settings (max tokens, temperature, top-k, top-p),
and generate a continuation.

Run locally:
    python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

This file is also the entry point for the Hugging Face Space (see app_file in README.md).
"""

import os
import torch
import tiktoken
import gradio as gr
from model import MiniGPT

# 1. Initialize identical model parameters
# These must match the values used in train.py, otherwise the saved weights won't load.
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
n_embd = 64
n_head = 4
n_layer = 4
block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 2. Instantiate model and load trained weights
model = MiniGPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)

if os.path.exists('mini_gpt.pt'):
    # map_location lets weights trained on GPU/MPS load on any device
    model.load_state_dict(torch.load('mini_gpt.pt', map_location=device, weights_only=True))
    print("Loaded trained weights successfully!")
else:
    print("Warning: mini_gpt.pt not found. Running with random weights.")

model.eval() # Set model to evaluation mode

# 3. The Generation Wrapper
def generate_text(prompt, max_tokens, temperature, top_k, top_p):
    """Called by Gradio when the button is clicked. Returns the prompt + generated continuation as text."""
    # Empty prompt: start from token id 0 ("!" in GPT-2 BPE). Otherwise tokenize the prompt.
    if not prompt.strip():
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        idx = torch.tensor([enc.encode(prompt, allowed_special="all")], dtype=torch.long, device=device)

    # Call your custom generation engine!
    # Sliders return floats, so cast to the types generate() expects.
    generated_indices = model.generate(
        idx,
        max_new_tokens=int(max_tokens),
        temperature=float(temperature),
        top_k=int(top_k),
        top_p=float(top_p)
    )

    # Token ids -> text (includes the original prompt)
    return enc.decode(generated_indices[0].tolist())

# 4. Build the Web Interface
with gr.Blocks(title="Mini-GPT Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Mini-GPT: Custom Decoder-Only Transformer")
    gr.Markdown("An autoregressive language model built from scratch in PyTorch, featuring an $O(1)$ KV-cache and a stochastic decoding engine.")

    with gr.Row():
        # Left column: decoding settings
        with gr.Column(scale=1):
            gr.Markdown("### Decoding Parameters")
            max_tokens = gr.Slider(10, 500, value=150, step=10, label="Max New Tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature", info="Higher = more random")
            top_k = gr.Slider(1, 50, value=10, step=1, label="Top-K", info="Limit to K most likely tokens")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P (Nucleus)", info="Cumulative probability cutoff")

        # Right column: prompt input, button, and output
        with gr.Column(scale=2):
            prompt = gr.Textbox(lines=4, placeholder="Enter a starting prompt (e.g., 'ROMEO: ')", label="Input Context")
            generate_btn = gr.Button("Generate Text", variant="primary")
            output = gr.Textbox(lines=10, label="Generated Output", interactive=False)

    # Wire the button to the function
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt, max_tokens, temperature, top_k, top_p],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
