from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch import optim
from tqdm import tqdm
from torch import amp

from data.datasets import create_dataloaders
from models.transformer import TransformerLanguageModel


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class WarmupCosineSchedule:
    optimizer: optim.Optimizer
    warmup: int
    total: int
    final_lr_ratio: float
    base_lr: float

    def step(self, step: int):
        if step < self.warmup:
            lr = self.base_lr * (step + 1) / max(1, self.warmup)
        else:
            progress = (step - self.warmup) / max(1, self.total - self.warmup)
            min_lr = self.base_lr * self.final_lr_ratio
            lr = min_lr + 0.5 * (self.base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


class SchedulerWrapper:
    """Unifies step interfaces for cosine (per-batch) and plateau (per-epoch)."""

    def __init__(self, kind: str, optimizer: optim.Optimizer, total_steps: int, tr_cfg):
        self.kind = kind
        self.optimizer = optimizer
        self.inner = None
        if kind == "cosine":
            self.inner = WarmupCosineSchedule(
                optimizer,
                warmup=int(tr_cfg.get("warmup_steps", 0)),
                total=total_steps,
                final_lr_ratio=float(tr_cfg.get("cosine_final_lr_ratio", 0.1)),
                base_lr=float(tr_cfg["learning_rate"]),
            )
        elif kind == "plateau":
            p = tr_cfg.get("plateau", {})
            self.inner = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=float(p.get("factor", 0.5)),
                patience=int(p.get("patience", 2)),
                min_lr=float(p.get("min_lr", 1e-5)),
            )

    def step_batch(self, step: int):
        if self.kind == "cosine" and self.inner is not None:
            self.inner.step(step)

    def step_epoch(self, val_loss: float):
        if self.kind == "plateau" and self.inner is not None:
            self.inner.step(val_loss)


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip, scaler, block_size: int, log_interval: int, grad_accum_steps: int = 1):
    model.train()
    total_loss = 0.0
    step = 0  # optimizer step count
    pbar = tqdm(loader, desc="train", leave=False)
    tokens_seen = 0
    from time import perf_counter
    t0 = perf_counter()
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        with amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        # gradient accumulation
        loss_to_backprop = loss / max(1, grad_accum_steps)
        if scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
            if grad_clip is not None and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            loss_to_backprop.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        micro += 1
        if micro % max(1, grad_accum_steps) == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step_batch(step)
            step += 1
        total_loss += loss.item() * x.size(0)
        tokens_seen += x.size(0) * x.size(1)
        if step % max(1, log_interval) == 0:
            dt = perf_counter() - t0
            tps = tokens_seen / max(1e-9, dt)
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "lr": f"{lr:.2e}", "tps": f"{tps/1e3:.1f}k/s"})
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, scaler):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        total_loss += loss.item() * x.size(0)
    avg = total_loss / len(loader.dataset)
    ppl = math.exp(min(20, avg))
    return avg, ppl


def plot_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="train", marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label="val", marker='o')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(cfg: Dict[str, Any]):
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("training", {}).get("device", "auto"))
    # Performance knobs
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    exp_name = cfg.get("experiment_name", "exp1")
    out_dir = os.path.join("results", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # Data (coerce basic types for robustness)
    data_cfg = cfg["data"]
    data_cfg["block_size"] = int(data_cfg["block_size"]) if isinstance(data_cfg["block_size"], (str, int)) else data_cfg["block_size"]
    if "train_fraction" in data_cfg:
        try:
            data_cfg["train_fraction"] = float(data_cfg["train_fraction"])
        except Exception:
            pass
    train_loader, val_loader, vocab = create_dataloaders(
        data_cfg["local_path"],
        block_size=data_cfg["block_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=(device.type == "cuda"),
        persistent_workers=None,
        train_fraction=data_cfg.get("train_fraction", 0.9),
    )

    # Model (ensure numeric fields are ints/floats)
    model_cfg = cfg["model"].copy()
    for k in ["d_model", "n_heads", "d_ff", "n_layers"]:
        if k in model_cfg and isinstance(model_cfg[k], str):
            model_cfg[k] = int(model_cfg[k])
    if "dropout" in model_cfg and isinstance(model_cfg["dropout"], str):
        model_cfg["dropout"] = float(model_cfg["dropout"])
    model_cfg["vocab_size"] = vocab.size
    model = TransformerLanguageModel(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)
    # Optional compile for PyTorch 2.x
    if cfg.get("training", {}).get("compile", False):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Optim + schedule (robust type coercion for YAML/CLI values)
    tr_cfg = cfg["training"]
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return x
    def _to_int(x):
        try:
            return int(x)
        except Exception:
            return x
    tr_cfg["learning_rate"] = _to_float(tr_cfg["learning_rate"])
    tr_cfg["weight_decay"] = _to_float(tr_cfg.get("weight_decay", 0.0))
    if "grad_clip" in tr_cfg and tr_cfg["grad_clip"] is not None:
        tr_cfg["grad_clip"] = _to_float(tr_cfg["grad_clip"])
    tr_cfg["warmup_steps"] = _to_int(tr_cfg.get("warmup_steps", 0))
    tr_cfg["cosine_final_lr_ratio"] = _to_float(tr_cfg.get("cosine_final_lr_ratio", 0.1))
    tr_cfg["batch_size"] = _to_int(tr_cfg["batch_size"])
    tr_cfg["max_epochs"] = _to_int(tr_cfg["max_epochs"])
    tr_cfg["num_workers"] = _to_int(tr_cfg.get("num_workers", 0))
    if isinstance(tr_cfg.get("betas"), (list, tuple)):
        tr_cfg["betas"] = tuple(_to_float(b) for b in tr_cfg["betas"])
    # Try fused AdamW on CUDA for higher throughput
    optimizer = None
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=tr_cfg["learning_rate"],
                betas=tr_cfg["betas"],
                weight_decay=tr_cfg["weight_decay"],
                fused=True,
            )
        except TypeError:
            pass
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tr_cfg["learning_rate"],
            betas=tr_cfg["betas"],
            weight_decay=tr_cfg["weight_decay"],
        )
    total_steps = tr_cfg["max_epochs"] * len(train_loader)
    scheduler_kind = str(tr_cfg.get("scheduler", "cosine")).lower()
    if scheduler_kind not in {"cosine", "plateau", "none"}:
        scheduler_kind = "cosine"
    scheduler = None if scheduler_kind == "none" else SchedulerWrapper(scheduler_kind, optimizer, total_steps, tr_cfg)

    train_losses, val_losses = [], []
    best_val = float("inf")
    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(tr_cfg.get("amp", True))))

    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,ppl,lr\n")

    no_improve = 0
    stop_file = tr_cfg.get("stop_file")
    eval_every = int(cfg.get("logging", {}).get("eval_every", 1))
    try:
        for epoch in range(1, tr_cfg["max_epochs"] + 1):
            tl = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                device,
                tr_cfg.get("grad_clip"),
                scaler,
                data_cfg["block_size"],
                cfg.get("logging", {}).get("log_interval", 100),
                grad_accum_steps=int(tr_cfg.get("grad_accum_steps", 1)),
            )
            do_eval = (epoch % eval_every == 0) or (epoch == tr_cfg["max_epochs"])  # always eval last epoch
            if do_eval:
                vl, ppl = evaluate(model, val_loader, device, scaler)
                if scheduler is not None:
                    scheduler.step_epoch(vl)
            else:
                vl, ppl = float("nan"), float("nan")
            train_losses.append(tl)
            val_losses.append(vl)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {epoch}: train_loss={tl:.3f} val_loss={vl:.3f} ppl={ppl:.2f} lr={current_lr:.2e}")
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{tl:.6f},{vl if not math.isnan(vl) else ''},{ppl if not math.isnan(vl) else ''},{current_lr:.8f}\n")

            if epoch % cfg["logging"].get("save_every", 1) == 0:
                save_checkpoint(model, optimizer, epoch, os.path.join(out_dir, "checkpoints", f"epoch_{epoch}.pt"))
                plot_curves(train_losses, val_losses, os.path.join(out_dir, "curves.png"))
            if not math.isnan(vl):
                improved = vl < best_val - float(cfg["training"].get("early_stopping", {}).get("min_delta", 0.0))
                if improved:
                    best_val = vl
                    save_checkpoint(model, optimizer, epoch, os.path.join(out_dir, "checkpoints", "best.pt"))
                    no_improve = 0
                else:
                    no_improve += 1
                # Early stopping
                es = cfg["training"].get("early_stopping", {})
                if bool(es.get("enabled", False)) and no_improve >= int(es.get("patience", 3)):
                    print(f"Early stopping triggered at epoch {epoch} (no improvement for {no_improve} epochs)")
                    with open(os.path.join(out_dir, "stopped_early.txt"), "w", encoding="utf-8") as f:
                        f.write(f"stopped_at_epoch: {epoch}\n")
                        f.write(f"best_val_loss: {best_val:.6f}\n")
                        f.write(f"patience: {int(es.get('patience',3))}\n")
                    break

            # Manual stop via file flag
            if stop_file and os.path.exists(stop_file):
                print(f"Stop file detected at {stop_file}. Stopping after epoch {epoch}.")
                with open(os.path.join(out_dir, "stopped_manual.txt"), "w", encoding="utf-8") as f:
                    f.write(f"stopped_at_epoch: {epoch}\n")
                    f.write(f"reason: stop_file\n")
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Saving interrupted checkpoint and exiting...")
        save_checkpoint(model, optimizer, epoch if 'epoch' in locals() else 0, os.path.join(out_dir, "checkpoints", "interrupted.pt"))

    # Save config copy
    with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries like key1.key2=value",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    # Apply overrides
    for item in args.override:
        k, v = item.split("=", 1)
        ref = cfg
        keys = k.split(".")
        for sub in keys[:-1]:
            ref = ref[sub]
        # Try parse number/bool
        if v.lower() in {"true", "false"}:
            val = v.lower() == "true"
        else:
            try:
                if "." in v:
                    val = float(v)
                else:
                    val = int(v)
            except ValueError:
                val = v
        ref[keys[-1]] = val
    return cfg


if __name__ == "__main__":
    cfg = parse_cli()
    main(cfg)
