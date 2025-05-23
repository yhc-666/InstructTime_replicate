"""Data loading utilities for AR and SFT training."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from multidataset import ARDataset, SFTDataset


# ---------------------------------------------------------------------------
# collate functions

def collate_fn_ar_train(batch):
    """Collate function for AR train/validation."""
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attn_masks"] for b in batch]
    label_ids = [b["label_ids"] for b in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label_ids": torch.stack(label_ids),
    }


# AR uses the same format for validation
collate_fn_ar_valid = collate_fn_ar_train


def collate_fn_sft_train(batch):
    """Collate function for SFT train/validation."""
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attn_masks"] for b in batch]
    label_ids = [b["label_ids"] for b in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label_ids": torch.stack(label_ids),
    }


collate_fn_sft_valid = collate_fn_sft_train


def collate_fn_sft_test(batch):
    """Collate function for SFT testing."""
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attn_masks"] for b in batch]
    labels = [b["label"] for b in batch]
    labels = torch.stack(labels)
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label": labels,
    }


# ---------------------------------------------------------------------------
# dataloader helpers

def build_ar_dataloaders(
    train_files: Sequence[str],
    val_files: Sequence[str],
    tokenizer,
    batch_size: int,
    encoder_max_length: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Construct dataloaders for AR training."""
    train_ds = ARDataset(train_files, tokenizer, mode="train", encoder_max_length=encoder_max_length)
    val_ds = ARDataset(val_files, tokenizer, mode="train", encoder_max_length=encoder_max_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_ar_train)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_ar_valid)
    return train_dl, val_dl


def build_sft_dataloaders(
    train_files: Sequence[str],
    val_files: Sequence[str],
    test_files: Sequence[str],
    tokenizer,
    batch_size: int,
    encoder_max_length: int,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct dataloaders for SFT training."""
    train_ds = SFTDataset(train_files, tokenizer, mode="train", encoder_max_length=encoder_max_length)
    val_ds = SFTDataset(val_files, tokenizer, mode="train", encoder_max_length=encoder_max_length)
    test_ds = SFTDataset(test_files, tokenizer, mode="test", encoder_max_length=encoder_max_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_sft_train)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_sft_valid)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_sft_test)
    return train_dl, val_dl, test_dl

