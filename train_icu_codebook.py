import os
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
import json
from TStokenizer.model import TStokenizer


class ICUTimeSeriesDataset(Dataset):
    """Dataset returning padded sequence, mask and valid length."""

    def __init__(self, root: str, split: str, max_len: int = 48, smoke_test: bool = False):
        self.data: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.lengths: List[int] = []
        self.max_len = max_len

        for task in ["ihm", "pheno"]:
            p = os.path.join(root, task, f"{split}_p2x_data.pkl")
            if not os.path.isfile(p):
                continue
            with open(p, "rb") as f:
                samples = pickle.load(f)
            
            # 只使用前30个样本进行smoke test
            if smoke_test:
                samples = samples[:30]
                
            for sample in samples:
                seq = np.asarray(sample["reg_ts"], dtype=np.float32)
                valid_len = min(len(seq), self.max_len)
                mask = np.zeros(self.max_len, dtype=np.float32)
                mask[:valid_len] = 1.0
                if len(seq) < self.max_len:
                    pad = np.zeros((self.max_len - len(seq), seq.shape[1]), dtype=np.float32)
                    seq = np.concatenate([seq, pad], axis=0)
                else:
                    seq = seq[:self.max_len]

                self.data.append(seq)
                self.masks.append(mask[:, None])
                self.lengths.append(valid_len)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return (
            torch.from_numpy(self.data[idx]),
            torch.from_numpy(self.masks[idx]),
            self.lengths[idx],
        )


def train_icu_codebook(
    data_root: str,
    save_dir: str = "./ecg_tokenizer/test_ecg_patch64",
    *,                            
    hidden_dim: int = 64,
    n_embed: int = 256,
    wave_length: int = 4,
    block_num: int = 4,
    max_len: int = 48,            # 数据集最大时间步
    ts_channels: int = 34,       # 时间序列通道数
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda",
    smoke_test: bool = False,
    use_wandb: bool = False
):

    """Train VQ-VAE codebook on IHM+PHE datasets."""

    train_set = ICUTimeSeriesDataset(data_root, "train",
                                     max_len=max_len, smoke_test=smoke_test)
    val_set   = ICUTimeSeriesDataset(data_root, "val",
                                     max_len=max_len, smoke_test=smoke_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    model = TStokenizer(
        data_shape=(max_len, ts_channels),
        hidden_dim=hidden_dim,
        n_embed=n_embed,
        block_num=block_num,
        wave_length=wave_length
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    hyper = dict(
        data_shape=(max_len, ts_channels),
        hidden_dim=hidden_dim,
        n_embed=n_embed,
        block_num=block_num,
        wave_length=wave_length
    )
    
    if use_wandb:
        wandb.init(project="icu-vq-codebook", config=hyper | dict(
        epochs=epochs, lr=lr, batch_size=batch_size,
        device=device, smoke_test=smoke_test))
        wandb.watch(model)

    for epoch in range(epochs):
        # ───────────── train ──────────────
        model.train()
        loss_sum = recon_sum = vq_sum = token_sum = 0.0
        used_ids = set()
        for seqs, mask, _ in train_loader:
            seqs, mask = seqs.to(device), mask.to(device)
            recon, diff, ids = model(seqs, mask=mask)

            recon_loss = ((recon - seqs) ** 2 * mask).sum() / mask.sum()
            loss = recon_loss + diff
            optim.zero_grad(); loss.backward(); optim.step()

            # 统计
            step_tokens = mask.sum().item()          # 有效时间步
            token_sum  += step_tokens
            loss_sum   += loss.item()   * step_tokens
            recon_sum  += recon_loss.item() * step_tokens
            vq_sum     += diff.item()   * step_tokens
            used_ids.update(ids.flatten().tolist())

        avg_train_loss  = loss_sum  / token_sum
        avg_train_recon = recon_sum / token_sum
        avg_train_vq    = vq_sum    / token_sum
        code_usage_train = len(used_ids) / model.n_embed   # 0-1

        # ───────────── valid ─────────────
        model.eval()
        loss_sum = recon_sum = vq_sum = token_sum = 0.0
        used_ids.clear()
        with torch.no_grad():
            for seqs, mask, _ in val_loader:
                seqs, mask = seqs.to(device), mask.to(device)
                recon, diff, ids = model(seqs, mask=mask)

                recon_loss = ((recon - seqs) ** 2 * mask).sum() / mask.sum()

                step_tokens = mask.sum().item()
                token_sum  += step_tokens
                loss_sum   += (recon_loss + diff).item() * step_tokens
                recon_sum  += recon_loss.item()          * step_tokens
                vq_sum     += diff.item()                * step_tokens
                used_ids.update(ids.flatten().tolist())

        avg_val_loss  = loss_sum  / token_sum
        avg_val_recon = recon_sum / token_sum
        avg_val_vq    = vq_sum    / token_sum
        code_usage_val = len(used_ids) / model.n_embed

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Train {avg_train_loss:.4f}  Val {avg_val_loss:.4f}  "
              f"CodeUsed {code_usage_train:.2%}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss":  avg_train_loss,
                "val_loss":    avg_val_loss,
                "train_recon": avg_train_recon,
                "val_recon":   avg_val_recon,
                "train_vq":    avg_train_vq,
                "val_vq":      avg_val_vq,
                "code_usage_train": code_usage_train,
                "code_usage_val":   code_usage_val,
            })
    
    # --------(1) 创建保存目录--------
    os.makedirs(save_dir, exist_ok=True)

    # --------(2) 把所有超参写入 args.json--------
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(hyper, f, indent=2)

    # --------(3) 保存纯 state-dict--------
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pkl"))

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse, pathlib, json

    parser = argparse.ArgumentParser(description="Train ICU VQ codebook")
    # 核心路径
    parser.add_argument("--data_root", type=str, required=True,
                        help="Folder with ihm/ and pheno/ pkl splits")
    parser.add_argument("--save_dir",  type=str,
                        default="TStokenizer/Vq_weight",
                        help="Directory to dump args.json & model.pkl")
    # 训练超参（可选）
    parser.add_argument("--hidden_dim",    type=int, default=64)
    parser.add_argument("--n_embed",    type=int, default=256)
    parser.add_argument("--wave_length",type=int, default=4)
    parser.add_argument("--block_num",  type=int, default=4)
    parser.add_argument("--max_len",    type=int, default=48)
    parser.add_argument("--ts_channels",type=int, default=34)
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)

    # 运行环境
    parser.add_argument("--device",     type=str, default="cpu")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--use_wandb",  action="store_true")

    args = parser.parse_args()

    train_icu_codebook(
        data_root   = args.data_root,
        save_dir    = args.save_dir,
        hidden_dim     = args.hidden_dim,
        n_embed     = args.n_embed,
        wave_length = args.wave_length,
        block_num   = args.block_num,
        max_len     = args.max_len,
        ts_channels = args.ts_channels,
        epochs      = args.epochs,
        lr          = args.lr,
        batch_size  = args.batch_size,
        device      = args.device,
        smoke_test  = args.smoke_test,
        use_wandb   = args.use_wandb,
    )


# /Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3
# /home/ubuntu/Virginia/output_mimic3

# python train_icu_codebook.py \
#     --data_root   /Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3 \
#     --smoke_test \
#     --epochs      1 \
#     --use_wandb




