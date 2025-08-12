# utils.py
import torch
import os
from config import cfg

def get_device():
    # Prefer MPS on macOS; fall back to CUDA then CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def save_checkpoint(path, model, optimizer, step, scaler=None, cfg=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
    }
    if scaler is not None:
        d["scaler_state"] = scaler.state_dict()
    if cfg is not None:
        d["cfg"] = cfg.__dict__
    torch.save(d, path)

def load_checkpoint(path, model, optimizer=None, scaler=None, device=None):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state"])
    if optimizer is not None and "optimizer_state" in ck:
        optimizer.load_state_dict(ck["optimizer_state"])
    if scaler is not None and "scaler_state" in ck:
        scaler.load_state_dict(ck["scaler_state"])
    step = ck.get("step", 0)
    return step
