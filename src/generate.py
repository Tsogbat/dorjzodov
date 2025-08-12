# generate.py
import argparse
import torch
from tokenizer import ByteTokenizer
from model import GPTByteModel
from config import cfg
from utils import get_device, load_checkpoint
import os

def run_generate(ckpt, prompt, max_new_tokens=128, temperature=1.0, greedy=False):
    device = get_device()
    cfg.device = str(device)
    model = GPTByteModel(cfg).to(device)
    load_checkpoint(ckpt, model, optimizer=None, scaler=None, device=device)
    model.eval()
    tk = ByteTokenizer()
    prompt_tokens = tk.encode(prompt)
    inp = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    if greedy:
        # temperature tiny and sample by top1
        out = model.generate(inp, max_new_tokens=max_new_tokens, temperature=1e-8)
    else:
        out = model.generate(inp, max_new_tokens=max_new_tokens, temperature=temperature)
    out_tokens = out[0].cpu().numpy().tolist()
    text = tk.decode(out_tokens)
    print("=== GENERATED ===")
    print(text)
    # write to outputs
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/generated.txt", "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Сайн уу")
    parser.add_argument("--max_new", type=int, default=128)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    run_generate(args.ckpt, args.prompt, args.max_new, args.temp, args.greedy)
