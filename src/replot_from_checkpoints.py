import argparse
import glob
import math
import os
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from data.datasets import create_dataloaders
from models.transformer import TransformerLanguageModel


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        total += loss.item() * x.size(0)
        n += x.size(0)
    avg = total / max(1, n)
    ppl = math.exp(min(20, avg))
    return avg, ppl


def main(exp_dir: str, eval_train: bool, eval_val: bool, device: str):
    cfg_path = os.path.join(exp_dir, "config_used.yaml")
    assert os.path.exists(cfg_path), f"Config not found: {cfg_path}"
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # Data
    train_loader, val_loader, vocab = create_dataloaders(
        cfg["data"]["local_path"],
        cfg["data"]["block_size"],
        cfg["training"]["batch_size"],
        num_workers=0,
        pin_memory=False,
        train_fraction=cfg["data"].get("train_fraction", 0.95),
    )

    # Model
    model = TransformerLanguageModel(
        vocab_size=vocab.size,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        d_ff=cfg["model"]["d_ff"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # Checkpoints
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    paths: List[str] = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
    assert paths, f"No epoch_*.pt checkpoints found in {ckpt_dir}"

    epochs, train_losses, val_losses = [], [], []
    metrics_csv = os.path.join(exp_dir, "metrics_from_ckpt.csv")
    with open(metrics_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_ppl\n")

    for p in paths:
        # extract epoch id
        base = os.path.basename(p)
        try:
            ep = int(base.split("_")[-1].split(".")[0])
        except Exception:
            continue
        state = torch.load(p, map_location=device)
        model.load_state_dict(state["model"])
        tl = 0.0
        if eval_train:
            tl, _ = evaluate(model, train_loader, device)
        vl, ppl = (0.0, 0.0)
        if eval_val:
            vl, ppl = evaluate(model, val_loader, device)
        epochs.append(ep)
        train_losses.append(tl if eval_train else float('nan'))
        val_losses.append(vl if eval_val else float('nan'))
        with open(metrics_csv, "a", encoding="utf-8") as f:
            f.write(f"{ep},{tl if eval_train else ''},{vl if eval_val else ''},{ppl if eval_val else ''}\n")

    # Plot
    plt.figure(figsize=(6, 4))
    if eval_train:
        plt.plot(epochs, train_losses, label="train", marker='o')
    if eval_val:
        plt.plot(epochs, val_losses, label="val", marker='o')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(exp_dir, "curves_from_ckpt.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Experiment directory under results/")
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    ap.add_argument("--eval", default="val", help="train|val|both")
    args = ap.parse_args()
    eval_train = args.eval in {"train", "both"}
    eval_val = args.eval in {"val", "both"}
    main(args.exp, eval_train=eval_train, eval_val=eval_val, device=args.device)

