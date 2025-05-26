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


# AR uses the same format for train and validation
collate_fn_ar_valid = collate_fn_ar_train


def collate_fn_sft_train(batch):
    """Collate function for SFT train"""
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attn_masks"] for b in batch]
    label_ids = [b["label_ids"] for b in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label_ids": torch.stack(label_ids),
    }


def collate_fn_sft_test(batch):
    """Collate function for SFT validation/testing."""
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attn_masks"] for b in batch]
    labels = [b["label"] for b in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label": torch.stack(labels),
    }

# SFT use the same format for validation and test
collate_fn_sft_valid = collate_fn_sft_test

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
    val_ds = SFTDataset(val_files, tokenizer, mode="test", encoder_max_length=encoder_max_length)
    test_ds = SFTDataset(test_files, tokenizer, mode="test", encoder_max_length=encoder_max_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_sft_train)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_sft_valid)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_sft_test)
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# 测试脚本

if __name__ == "__main__":
    import os
    import argparse
    from utils import load_TStokenizer
    from multimodel import MultiTokenizer
    
    parser = argparse.ArgumentParser(description="测试 DataLoader 内容")
    parser.add_argument("--mode", type=str, default="sft", choices=["ar", "sft"], help="选择测试模式")
    parser.add_argument("--dataset", type=str, default="ihm", choices=["ihm", "pheno", "mix"], help="选择数据集")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--num_batches", type=int, default=2, help="要查看的批次数量")
    args = parser.parse_args()
    
    # 设置路径
    vqvae_path = "TStokenizer/Vq_weight"
    DATA_ROOT = "ts_tokenized_datasets"
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    ts_tokenizer = load_TStokenizer(vqvae_path, data_shape=(48, 34), device="cpu")
    tokenizer = MultiTokenizer([ts_tokenizer])
    
    # 准备数据文件路径
    if args.dataset == "mix":
        train_files = [
            os.path.join(DATA_ROOT, "ihm", "tokenized_v1_patch64", "train.pkl"),
            os.path.join(DATA_ROOT, "pheno", "tokenized_v1_patch64", "train.pkl"),
        ]
        val_files = [
            os.path.join(DATA_ROOT, "ihm", "tokenized_v1_patch64", "val.pkl"),
            os.path.join(DATA_ROOT, "pheno", "tokenized_v1_patch64", "val.pkl"),
        ]
        test_files = [
            os.path.join(DATA_ROOT, "ihm", "tokenized_v1_patch64", "test.pkl"),
            os.path.join(DATA_ROOT, "pheno", "tokenized_v1_patch64", "test.pkl"),
        ]
    else:
        train_files = [os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "train.pkl")]
        val_files = [os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "val.pkl")]
        test_files = [os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "test.pkl")]
    
    # 构建 DataLoader
    print(f"\n构建 {args.mode.upper()} DataLoader...")
    
    if args.mode == "ar":
        train_dl, val_dl = build_ar_dataloaders(
            train_files, val_files, tokenizer, 
            batch_size=args.batch_size, 
            encoder_max_length=1024,
            num_workers=0  # 测试时设为0避免多进程问题
        )
        
        print("\n=== AR 训练 TrainDataLoader 内容 ===")
        print(train_dl.dataset[0]["input_ids"])
        print(train_dl.dataset[0]["attn_masks"])
        print(train_dl.dataset[0]["label_ids"])
        # 过滤掉padding token再解码
        valid_label_ids = train_dl.dataset[0]["label_ids"][train_dl.dataset[0]["label_ids"] != tokenizer.pad_token_id]
        print("label_ids对应的文字:", tokenizer.decode(valid_label_ids))
    
    else:  # sft mode
        train_dl, val_dl, test_dl = build_sft_dataloaders(
            train_files, val_files, test_files, tokenizer,
            batch_size=args.batch_size,
            encoder_max_length=1024,
            num_workers=0
        )
        
        print("\n=== SFT 训练 TrainDataLoader 内容 ===")
        print(train_dl.dataset[0]["input_ids"])
        print(train_dl.dataset[0]["attn_masks"])
        print(train_dl.dataset[0]["label_ids"])
        # 过滤掉-100和padding token再解码
        label_ids = train_dl.dataset[0]["label_ids"]
        valid_label_ids = label_ids[(label_ids != -100) & (label_ids != tokenizer.pad_token_id)]
        print("label_ids对应的文字:", tokenizer.decode(valid_label_ids))
        
        print("\n=== SFT 验证 ValidationDataLoader 内容 ===")
        print("input_ids (仅问题部分):", val_dl.dataset[0]["input_ids"])
        print("attn_masks:", val_dl.dataset[0]["attn_masks"])
        print("label (原始标签):", val_dl.dataset[0]["label"])
    
    print("\n测试完成")


# python datamodules.py --mode ar --dataset mix --batch_size 2 --num_batches 2
# python datamodules.py --mode sft --dataset ihm --batch_size 2 --num_batches 2



