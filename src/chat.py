# chat.py
import argparse
import torch
from tokenizer import ByteTokenizer
from model import GPTByteModel
from config import cfg
from utils import get_device, load_checkpoint

def chat_loop(ckpt):
    device = get_device()
    cfg.device = str(device)
    model = GPTByteModel(cfg).to(device)
    load_checkpoint(ckpt, model, optimizer=None, scaler=None, device=device)
    model.eval()
    tk = ByteTokenizer()
    print("Interactive chat. Type 'exit' to quit.")
    history = ""
    while True:
        usr = input("You: ")
        if usr.strip().lower() in ("exit", "quit"):
            break
        # simple system: concat history + user
        prompt = history + "\nUser: " + usr + "\nBot: "
        tokens = tk.encode(prompt)
        inp = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(inp, max_new_tokens=128, temperature=0.9)
        out_tokens = out[0].cpu().numpy().tolist()
        gen = tk.decode(out_tokens[len(tokens):])  # just the new content
        print("Bot:", gen)
        # append conversation
        history = (history + "\nUser: " + usr + "\nBot: " + gen)[-2000:]  # trim length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    chat_loop(args.ckpt)
