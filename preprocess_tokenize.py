"""
使用训练好的VQVAE TS tokenizer提前将数据集tokenize为token ids, 并保存为pkl文件, 加速后续训练

pipeline:

          ihm/train_p2x_data.pkl
                ▼           （同理 val / test；pheno 同理）
   ┌─────────────────────────────┐
   │  reg_ts  (T,34)             │
   │  text_data  (N notes)       │
   │  label                      │
   └─────────────────────────────┘
             │ ① 取最后 5 条 note
             │
   ┌─────────▼───────────┐
   │   VQ-VAE.encode()   │ ② padding→mask→index
   └─────────┬───────────┘
             │ ids : [tok1 … tokK]
             │
   ┌─────────▼───────────┐
   │ dict(ts_ids, notes, │
   │      label, name)   │
   └─────────┬───────────┘
             │
datasets/ihm/tokenized_v1_patch64/train.pkl ③ 写入

"""

# preprocess_tokenize.py
import os, pickle, json, math
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from TStokenizer.model import TStokenizer          
from utils import load_TStokenizer
import tqdm


def encode_ts(model: TStokenizer,
              ts: np.ndarray,
              max_len: int,
              device: str = "cpu") -> List[int]:
    """
    功能: 将单条时间序列数据转换为token ids
    输入:
        - ts: 时间序列数据, shape=(T, 34)
        - max_len: 最大长度, 48
    输出:
        - 长度为ceil(T / patch)的token ids列表
    """
    # 先把所有输入TS 统一pad到max_len
    T, C = ts.shape
    valid_len = min(T, max_len) # 24 for pheno, 48 for ihm
    if T < max_len:
        pad = np.zeros((max_len - T, C), dtype=np.float32)
        ts = np.concatenate([ts, pad], 0)
    else:
        ts = ts[:max_len]

    # 转换为torch.Tensor, 并添加batch维度
    ts_t = torch.from_numpy(ts).unsqueeze(0).to(device)        # (1, L, C)

    # 创建掩码, 用于mask掉padding部分
    mask = torch.zeros(1, max_len, 1, device=device)
    mask[0, :valid_len] = 1

    with torch.no_grad():
        _, _, ids = model(ts_t, mask=mask)     # ids : (1, L//patch)
    patch = model.wave_patch[0]
    valid_tokens = math.ceil(valid_len / patch)
    return ids[0, :valid_tokens].cpu().tolist()


def process_split(
        pkl_path: str,
        task: str,
        model: TStokenizer,
        max_len: int,
        keep_notes: int = 5,
        device: str = "cpu",
        smoke_test: bool = False,
        ) -> List[Dict[str, Any]]:
    """
    完整tokenize一个raw pkl文件并存储为new pkl文件
    """

    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)

    if smoke_test:                         # ← 只留前 N 条
        samples = samples[:30]

    out = []
    for s in tqdm.tqdm(samples, desc="处理样本"):
        ts_ids = encode_ts(model, s["reg_ts"].astype("float32"),
                           max_len=max_len, device=device)
        notes  = s["text_data"][-keep_notes:]   # 最后 N 条
        # 根据task选择label
        if task == "ihm":
            out.append(dict(
                ts_ids = ts_ids,
                notes  = notes,
                label  = s["label"],
                name   = s.get("name", "")
            ))
        elif task == "pheno":    
            out.append(dict(
                ts_ids = ts_ids,
                notes  = notes,
                label  = s["label"][1:],   # 之前预处理的时候label那边有问题, 把patient id也加进去了
                name   = s.get("name", "")
            ))
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("tokenize IHM / Pheno reg_ts")
    parser.add_argument("--data_root",   required=True, help="原始 p2x pkl 根目录")
    parser.add_argument("--tokenizer",   required=True, help="保存 args.json/model.pkl 的文件夹")
    parser.add_argument("--out_dir",     default="ts_tokenized_datasets", help="输出根目录")
    parser.add_argument("--keep_notes",  type=int, default=5, help="每个样本保留的note数量")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--smoke_test",  action="store_true", help="仅处理每个 split 的前 30 条样本，用于快速调试")
    args = parser.parse_args()

    cfgs = {    
        "ihm":   dict(max_len=48),
        "pheno": dict(max_len=48),
    }

    tokenizer = load_TStokenizer(args.tokenizer,
                                 data_shape=(48, 34),      # shape 与训练时保持一致
                                 device=args.device)

    for task, cfg in cfgs.items():
        print(f"\n=== {task.upper()} ===")
        task_in = Path(args.data_root) / task
        task_out = Path(args.out_dir) / task / "tokenized_v1_patch64"
        task_out.mkdir(parents=True, exist_ok=True)

        for split in ["train", "val", "test"]:
            pkl_in  = task_in / f"{split}_p2x_data.pkl"
            if not pkl_in.exists():
                print(f"skip {pkl_in} (not found)")
                continue
            print(f"→ {pkl_in.name}")

            data = process_split(str(pkl_in), task=task, model=tokenizer, 
                                 max_len=cfg["max_len"],
                                 keep_notes=args.keep_notes,
                                 device=args.device,
                                 smoke_test=args.smoke_test)

            pkl_out = task_out / f"{split}.pkl"
            with open(pkl_out, "wb") as f:
                pickle.dump(data, f)
            print(f"   write {len(data):,} samples  ➟  {pkl_out}")

    print("\nDone ✅")



# /Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3

# python preprocess_tokenize.py \
#    --data_root   /Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3 \
#    --tokenizer   TStokenizer/Vq_weight \
#    --device      cpu \
#    --smoke_test
