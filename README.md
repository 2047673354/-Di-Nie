# Transformer From Scratch (Small-Scale)

This repo implements a minimal Transformer (encoder and optional decoder) from scratch in PyTorch. It trains on a small text corpus for language modeling and produces reproducible curves and checkpoints.

Key features
- Scaled dot-product attention, multi-head attention, position-wise FFN
- Residual connections + LayerNorm (pre-norm), sinusoidal positional encoding
- Padding and subsequent masks, AdamW, gradient clipping, cosine LR schedule
- Character-level language modeling dataset for offline runs; easy to swap to public datasets later
- Reproducible training with fixed seeds; checkpoints and plots under `results/`

## Quick Start

1) Create environment (example with conda)

```
conda create -n transformer python=3.10 -y
conda activate transformer
pip install -r requirements.txt
```

2) Prepare a small text file. By default we expect `data/tiny_shakespeare.txt` (already provided). If missing, download from Karpathy's repo and save as `data/tiny_shakespeare.txt`.

```
mkdir -p data
# Put any plain text into data/tiny.txt (e.g., a few MB of public text)
```

3) Train with the default config

```
python src/train.py --config configs/base.yaml
```

Or via helper script:

```
sh scripts/run.sh
```

Outputs
- Checkpoints: `results/exp1/checkpoints/epoch_*.pt`
- Curves: `results/exp1/curves.png`
- Config copy + logs in the same folder

## Hardware Requirements

- GPU run (recommended): NVIDIA GPU with 6–8 GB VRAM for `configs/tiny_shakespeare_gpu.yaml` (batch 64). For 4 GB cards, use `--override training.batch_size=32 training.grad_accum_steps=2`.
- CPU run: works but is slower (tens of minutes per epoch). Use `configs/tiny_shakespeare_cpu.yaml`.

## Datasets

The default `CharDataset` reads a local text file and builds a character vocabulary. To use public datasets like WikiText-2 or Tiny Shakespeare from Hugging Face, see `src/data/datasets.py` and set `data.source=huggingface`. Internet is required to auto-download.

## Configuration

See `configs/base.yaml` for hyperparameters (embedding size, heads, layers, batch size, learning rate, etc.). Override via CLI, e.g. `--override model.n_heads=8 training.batch_size=64`.

## Reproducibility

- All random seeds fixed by `seed` in config.
- For strict reproducibility across GPUs, disable AMP/TF32 and keep the same driver/CUDA/PyTorch stack. Minor numeric drift may still occur across hardware.

### Exact Commands (with seed)

- GPU (baseline, 10 epochs, logs + checkpoints):

```
python src/train.py --config configs/tiny_shakespeare_gpu.yaml \
  --override experiment_name=paper_gpu seed=42
```

- CPU（较慢，但可复现）:

```
python src/train.py --config configs/tiny_shakespeare_cpu.yaml \
  --override experiment_name=paper_cpu seed=42
```

- Strict(er) mode（尽量一致的数值；关闭 AMP/编译/TF32）：

```
python src/train.py --config configs/tiny_shakespeare_gpu.yaml \
  --override experiment_name=paper_strict seed=42 training.amp=false training.compile=false
```

Producing results:
- Curves: `results/<experiment>/curves.png`
- Metrics: `results/<experiment>/metrics.csv`
- Best checkpoint: `results/<experiment>/checkpoints/best.pt`

### Optional: Manual Stop & Resume

- Graceful stop at epoch end: start with `--override training.stop_file=results/<exp>/STOP` and create that file to stop.
- Keyboard interrupt saves `checkpoints/interrupted.pt` before exit.

## Project Structure

```
src/
  models/transformer.py   # Transformer blocks and full model
  data/datasets.py        # Character LM dataset + optional HF datasets
  train.py                # Training/eval loop and plotting
configs/
  base.yaml               # Default hyperparameters
scripts/
  run.sh                  # One-liner to launch training
results/                  # Logs, checkpoints, curves (created on run)
```

## Citation

Please cite the original Transformer paper when writing the report:

Ashish Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
