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
    save_dir: str = "./icu_tokenizer",
    *,
    hidden_dim: int = 64,
    n_embed: int = 256,
    wave_length: int = 4,
    block_num: int = 4,
    max_len: int = 48,
    ts_channels: int = 34,
    epochs: int = 30,
    lr: float = 2e-4,
    batch_size: int = 128,
    device: str = "cuda",
    smoke_test: bool = False,
    use_wandb: bool = False,
    patience: int = 10,
    lr_decay_factor: float = 0.5,
    lr_decay_patience: int = 5,
    weight_decay: float = 1e-5,
):

    # -------- datasets --------
    train_set = ICUTimeSeriesDataset(data_root, "train",
                                     max_len=max_len, smoke_test=smoke_test)
    val_set   = ICUTimeSeriesDataset(data_root, "val",
                                     max_len=max_len, smoke_test=smoke_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    # -------- model / opt --------
    model = TStokenizer(
        data_shape=(max_len, ts_channels),
        hidden_dim=hidden_dim,
        n_embed=n_embed,
        block_num=block_num,
        wave_length=wave_length
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=lr_decay_factor,
        patience=lr_decay_patience, verbose=True)

    if use_wandb:
        wandb.init(project="icu-vq-codebook",
                   config=dict(hidden_dim=hidden_dim, n_embed=n_embed,
                               wave_length=wave_length, lr=lr, batch_size=batch_size,
                               epochs=epochs))
        wandb.watch(model)

    # -------- early-stop vars --------
    best_val = float("inf")
    best_state = None
    stale = 0

    # ===== main loop =====
    for ep in range(1, epochs + 1):

        # ----- train -----
        model.train()
        step_token_sum = step_patch_sum = 0.0
        recon_acc = vq_acc = 0.0
        used_ids = set()

        for seqs, mask, _ in train_loader:
            seqs, mask = seqs.to(device), mask.to(device)
            recon, diff, ids = model(seqs, mask=mask)

            # losses
            recon_loss = ((recon - seqs) ** 2 * mask).sum() / mask.sum()
            loss = recon_loss + diff

            optim.zero_grad()
            loss.backward()
            optim.step()

            # stats
            tok = mask.sum().item()
            pat = mask.view(mask.size(0), -1, wave_length).any(-1).sum().item()

            step_token_sum += tok
            step_patch_sum += pat
            recon_acc      += recon_loss.item() * tok
            vq_acc         += diff.item() * pat
            used_ids.update(ids.flatten().tolist())

        train_recon = recon_acc / step_token_sum
        train_vq    = vq_acc    / step_patch_sum
        train_loss  = train_recon + train_vq
        code_use_tr = len(used_ids) / n_embed

        # ----- valid -----
        model.eval()
        step_token_sum = step_patch_sum = 0.0
        recon_acc = vq_acc = 0.0
        used_ids.clear()

        with torch.no_grad():
            for seqs, mask, _ in val_loader:
                seqs, mask = seqs.to(device), mask.to(device)
                recon, diff, ids = model(seqs, mask=mask)

                recon_loss = ((recon - seqs) ** 2 * mask).sum() / mask.sum()

                tok = mask.sum().item()
                pat = mask.view(mask.size(0), -1, wave_length).any(-1).sum().item()

                step_token_sum += tok
                step_patch_sum += pat
                recon_acc      += recon_loss.item() * tok
                vq_acc         += diff.item() * pat
                used_ids.update(ids.flatten().tolist())

        val_recon = recon_acc / step_token_sum
        val_vq    = vq_acc    / step_patch_sum
        val_loss  = val_recon + val_vq
        code_use_val = len(used_ids) / n_embed

        # -- scheduler / early-stop --
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            stale = 0
            flag = "✓"
        else:
            stale += 1
            flag = " "

        print(f"{flag} Epoch {ep}/{epochs}  "
              f"Train {train_loss:.3f}  Val {val_loss:.3f}  "
              f"CodeUse {code_use_tr:.1%}  LR {optim.param_groups[0]['lr']:.1e}  "
              f"Pat {stale}/{patience}")

        if use_wandb:
            wandb.log(dict(epoch=ep,
                           train_loss=train_loss, val_loss=val_loss,
                           train_recon=train_recon, val_recon=val_recon,
                           train_vq=train_vq,   val_vq=val_vq,
                           code_usage_train=code_use_tr, code_usage_val=code_use_val,
                           lr=optim.param_groups[0]['lr']))

        if stale >= patience:
            print(f"Early-stop at epoch {ep}")
            break

    # -------- save --------
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state or model.state_dict(),
               os.path.join(save_dir, "model.pkl"))
    with open(os.path.join(save_dir, "args.json"), "w") as fp:
        json.dump(hyper := dict(hidden_dim=hidden_dim, n_embed=n_embed,
                                wave_length=wave_length, block_num=block_num), fp, indent=2)

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
    parser.add_argument("--wave_length",type=int, default=2)
    parser.add_argument("--block_num",  type=int, default=4)
    parser.add_argument("--max_len",    type=int, default=48)
    parser.add_argument("--ts_channels",type=int, default=34)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=4e-5)
    parser.add_argument("--batch_size", type=int, default=128)

    # 运行环境
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--use_wandb",  action="store_true")
    
    # 新增正则化参数
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--lr_decay_factor", type=float, default=0.5,
                        help="Learning rate decay factor")
    parser.add_argument("--lr_decay_patience", type=int, default=5,
                        help="Learning rate decay patience")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for regularization")

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
        patience    = args.patience,
        lr_decay_factor = args.lr_decay_factor,
        lr_decay_patience = args.lr_decay_patience,
        weight_decay = args.weight_decay,
    )


# /Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3
# /home/ubuntu/Virginia/output_mimic3
# /home/ubuntu/hcy50662/output_mimic3


# 改进的训练命令（解决过拟合问题）
# 1. 使用完整数据集训练（移除 --smoke_test）
# python train_icu_codebook.py \
#     --data_root   /home/ubuntu/hcy50662/output_mimic3 \
#     --use_wandb

# 2. 如果必须使用smoke_test，增加正则化
# python train_icu_codebook.py \
#     --data_root   /home/ubuntu/hcy50662/output_mimic3 \
#     --smoke_test \
#     --epochs      30 \
#     --lr          1e-4 \
#     --batch_size  8 \
#     --patience    10 \
#     --lr_decay_factor 0.5 \
#     --lr_decay_patience 5 \
#     --weight_decay 1e-3 \
#     --use_wandb

# 3. 更保守的训练设置
# python train_icu_codebook.py \
#     --data_root   /home/ubuntu/hcy50662/output_mimic3 \
#     --epochs      100 \
#     --lr          1e-4 \
#     --batch_size  16 \
#     --patience    20 \
#     --lr_decay_factor 0.8 \
#     --lr_decay_patience 10 \
#     --weight_decay 5e-4 \
#     --n_embed     512 \
#     --hidden_dim  128 \
#     --use_wandb




