"""
作用: TStokenizer训练的主入口脚本，基于VQVAE模型将时间序列数据转化为离散tokens
输入: 
    - 预处理后的时间序列数据
    - 模型参数(命令行参数指定)
输出: 
    - 训练好的TStokenizer模型
    - 训练和评估指标记录
示例CLI:
    python main.py \
    --save_path "./ecg_tokenizer" \
    --dataset "ecg" \
    --data_path "./data/ecg_data" \
    --device "cuda:0" \
    --d_model 64 \
    --wave_length 32 \
    --n_embed 128
"""

import os
import torch
import random
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from dataset import Dataset
from args import args
from process import Trainer
from model import VQVAE
import torch.utils.data as Data

def seed_everything(seed):
    """
    功能: 设置所有随机种子，确保实验可重复性
    输入:
        - seed: 整数随机种子
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def main():
    """
    功能: 主函数，执行VQVAE训练流程
    流程:
        1. 设置随机种子
        2. 加载训练和测试数据集
        3. 初始化VQVAE模型
        4. 创建训练器对象
        5. 执行训练过程
    """
    # 设置随机种子，确保实验可重复性
    seed_everything(seed=2023)

    # 加载训练数据集
    train_dataset = Dataset(device=args.device, mode='train', args=args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    
    # 加载验证数据集
    val_dataset = Dataset(device=args.device, mode='test', args=args)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.test_batch_size)
    print(args.data_shape)
    print('dataset initial ends')

    # 初始化VQVAE模型
    # - data_shape: 输入时间序列形状 (seq_len, feat_dim)
    # - hidden_dim: 隐藏层维度
    # - n_embed: codebook大小，即离散码本中的码字数量
    # - block_num: TCN块的数量
    # - wave_length: 时间窗口大小，控制压缩比例
    model = VQVAE(data_shape=args.data_shape, hidden_dim=args.d_model, n_embed=args.n_embed, block_num=args.block_num,
                    wave_length=args.wave_length)
    print('model initial ends')

    # 创建训练器对象，管理训练和评估流程
    trainer = Trainer(args, model, train_loader, val_loader, verbose=True)
    print('trainer initial ends')

    # 开始训练过程
    trainer.train()


if __name__ == '__main__':
    main()

