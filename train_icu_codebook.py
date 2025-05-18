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
    save_path: str = "vq.ckpt",
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str = "cuda",
    smoke_test: bool = False,
    use_wandb: bool = False
    ):
    """Train VQ-VAE codebook on IHM+PHE datasets."""

    train_set = ICUTimeSeriesDataset(data_root, "train", smoke_test=smoke_test)
    val_set = ICUTimeSeriesDataset(data_root, "val", smoke_test=smoke_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = TStokenizer(
        data_shape=(48, 34), hidden_dim=64, n_embed=256, block_num=4, wave_length=4
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    if use_wandb:
        wandb.init(project="icu-vq-codebook", config={
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "device": device,
            "smoke_test": smoke_test
        })
        wandb.watch(model)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for seqs, mask, _ in train_loader:
            seqs = seqs.to(device)
            mask = mask.to(device)
            recon, diff, _ = model(seqs, mask=mask)
            recon_loss = ((recon - seqs) ** 2 * mask).sum() / mask.sum()
            loss = recon_loss + diff
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
            
        avg_train_loss = np.mean(train_losses)
        
        # simple val loss print
        model.eval()
        with torch.no_grad():
            losses = []
            for seqs, mask, _ in val_loader:
                seqs = seqs.to(device)
                mask = mask.to(device)
                recon, diff, _ = model(seqs, mask=mask)
                recon_loss = ((recon - seqs) ** 2 * mask).sum() / mask.sum()
                losses.append((recon_loss + diff).item())
        avg_val_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "recon_loss": recon_loss.item(),
                "vq_diff": diff.item()
            })

    torch.save(model.state_dict(), save_path)
    
    if use_wandb:
        wandb.finish()


class ICUCodebook(nn.Module):
    """Utility wrapper providing ``encode`` for trained codebook."""

    def __init__(self, ckpt_path: str, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.model = TStokenizer(data_shape=(48, 34), hidden_dim=64, n_embed=256,
                                 block_num=4, wave_length=4)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, ts: np.ndarray, valid_len: int | None = None) -> np.ndarray:
        """Convert ``(T,34)`` array into token indices.

        Parameters
        ----------
        ts : np.ndarray
            Time series with shape ``(T, 34)``.
        valid_len : int, optional
            Actual sequence length. If ``None``, use ``ts.shape[0]``.
        """
        arr = np.asarray(ts, dtype=np.float32)
        valid_len = arr.shape[0] if valid_len is None else valid_len
        arr = arr[:valid_len]
        if arr.shape[0] < 48:
            pad = np.zeros((48 - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        mask = torch.zeros(1, 48, 1, device=self.device)
        mask[0, :valid_len] = 1
        _, _, ids = self.model(tensor, mask=mask)
        patch_len = self.model.wave_patch[0]
        max_tokens = ids.shape[1]
        valid_tokens = (valid_len + patch_len - 1) // patch_len
        return ids[:, :valid_tokens].squeeze(0).cpu().numpy()


__all__ = ["ICUTimeSeriesDataset", "train_icu_codebook", "ICUCodebook"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ICU VQ codebook")
    parser.add_argument("--data_root", type=str, default="/Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3",
                        help="Path containing ihm/ and pheno/ folders")
    parser.add_argument("--save_path", type=str, default="TStokenizer/Vq_weight/vq.ckpt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke_test", action="store_true",
                        help="开启后使用每个pkl文件的前30个样本进行快速测试")
    parser.add_argument("--use_wandb", action="store_true",
                        help="启用wandb记录训练进度")
    args = parser.parse_args()

    train_icu_codebook(args.data_root, save_path=args.save_path,
                       epochs=args.epochs, lr=args.lr, device=args.device,
                       smoke_test=args.smoke_test, use_wandb=args.use_wandb)
    
# smoke test:
# python train_icu_codebook.py --data_root 您的数据路径 --smoke_test --use_wandb
