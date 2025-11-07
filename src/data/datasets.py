from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class Vocab:
    stoi: dict
    itos: List[str]

    @classmethod
    def build_from_text(cls, text: str) -> "Vocab":
        chars = sorted(list(set(text)))
        itos = ["<pad>"] + chars  # pad at index 0
        stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        stoi["<pad>"] = 0
        return cls(stoi=stoi, itos=itos)

    @property
    def size(self) -> int:
        return len(self.itos)


class CharDataset(Dataset):
    """Character-level language modeling dataset built from a local text file.

    Generates overlapping blocks of length `block_size` where the target is the next character.
    """

    def __init__(self, path: str, block_size: int, split: str = "train", train_fraction: float = 0.9, vocab: Optional[Vocab] = None):
        super().__init__()
        assert os.path.exists(path), f"Text file not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        n = int(len(text) * train_fraction)
        if split == "train":
            text = text[:n]
        else:
            text = text[n:]

        # Build or use provided vocab (always use train vocab for val for consistency)
        self.vocab = vocab if vocab is not None else Vocab.build_from_text(text)
        self.block_size = block_size
        self.data = [self.vocab.stoi.get(ch, 0) for ch in text]

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1 : idx + 1 + self.block_size], dtype=torch.long)
        return x, y


# NOTE: top-level collate for Windows DataLoader (picklable)
def collate_char(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.stack([b[1] for b in batch], dim=0)
    return x, y


def create_dataloaders(
    path: str,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    train_fraction: float = 0.9,
) -> Tuple[DataLoader, DataLoader, Vocab]:
    train_ds = CharDataset(path, block_size, split="train", train_fraction=train_fraction)
    # Encode validation using the training vocabulary to avoid index mismatch
    val_ds = CharDataset(path, block_size, split="val", train_fraction=train_fraction, vocab=train_ds.vocab)

    def collate(batch):
        x = torch.stack([b[0] for b in batch], dim=0)
        y = torch.stack([b[1] for b in batch], dim=0)
        return x, y

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_char,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    return train_loader, val_loader, train_ds.vocab


__all__ = ["CharDataset", "create_dataloaders", "Vocab"]
