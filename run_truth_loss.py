"""
作用: 
    - InstructTime模型训练和推理的主入口脚本, 实现了训练、验证和测试流程
    - 能同时进行两种训练模式 1.训练Universal模型(Auto-Regressive) 2.训练Adapt模型(Supervised Fine-Tuning)
输入: 
    - 预处理后的数据
    - 预训练的GPT2模型
    - 已训练好的 VQVAE tokenizer
输出: 
    - 训练好的InstructTime模型
    - 评估结果和预测
示例CLI:
    # 训练Universal模型(Auto-Regressive)
    python run_truth_loss.py --device "cuda:0" --dataset "mix" --batch_size 16 --lr 1e-5 --epochs 15 --adapt False
    
    # 训练Adapt模型(SFT)
    python run_truth_loss.py --device "cuda:0" --dataset "ecg" --batch_size 16 --lr 1e-5 --epochs 10 --adapt True --load_model_path "./gptmodel"
"""

import os
import torch
import random
import logging
from logging.handlers import RotatingFileHandler
import pickle
import transformers
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
)
import wandb

from multimodel import InstructTime, MultiTokenizer
from multidataset import MultiDataset
from args import get_hyperparams
from metrics import metric_ecg, metric_eeg, metric_har, metric_fd, metric_rwc
from utils import extract_all_information, load_TStokenizer

local_model_path = "./gpt2-model"
# 统一的 ICU VQ-VAE tokenizer 路径
vqvae_path = "TStokenizer/Vq_weight"
# tokenized dataset 根目录
DATA_ROOT = "ts_tokenized_datasets"

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

def collate_fn_train(batch):
    """
    功能: 训练数据批处理函数，收集批次中的样本并组织为模型输入格式
    输入:
        - batch: 批次数据列表，每个元素为字典，包含input_ids, attn_masks, label_ids
    输出:
        - 组织好的批次字典，包含:
            - input_ids: 输入ID张量, shape=[batch_size, seq_len]
            - attention_mask: 注意力掩码张量, shape=[batch_size, seq_len]
            - label_ids: 标签ID张量, shape=[batch_size, seq_len]
    """
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label_ids": torch.stack(label_ids),
    }

def collate_fn_validation(batch):
    """
    功能: 验证数据批处理函数，收集批次中的样本并组织为模型输入格式
    输入:
        - batch: 批次数据列表，每个元素为字典，包含input_ids, attn_masks, label
    输出:
        - 组织好的批次字典，包含:
            - input_ids: 输入ID张量, shape=[batch_size, seq_len]
            - attention_mask: 注意力掩码张量, shape=[batch_size, seq_len]
            - labels: 原始标签文本列表, 每个元素为字符串
    """
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    labels = [x["label"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": labels,
    }

def collate_fn_test(batch):
    """
    功能: 测试数据批处理函数，收集批次中的样本并组织为模型输入格式
    输入:
        - batch: 批次数据列表，每个元素为字典，包含input_ids, attn_masks, label
    输出:
        - 组织好的批次字典，包含:
            - input_ids: 输入ID张量, shape=[batch_size, seq_len]
            - attention_mask: 注意力掩码张量, shape=[batch_size, seq_len]
            - labels: 原始标签文本列表, 每个元素为字符串
    """
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    labels = [x["label"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": labels,
    }



def validate_sft(model, ValidDataLoader, args, logger, out=False):
    """
    功能: 验证SFT模型在验证集上的性能
    输入:
        - model: InstructTime模型实例
    """
    pass


def validate_ar(model, ValidDataLoader, args, logger, out=False):
    """
    功能: 验证AR模型在验证集上的性能
    输入:
        - model: InstructTime模型实例
    """
    model.eval()
    loss_sum = 0.0
    step = 0
    with torch.no_grad():
        for data in tqdm(ValidDataLoader, desc="Val", ncols=120):
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            label_ids = data["label_ids"].to(args.device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids,
                )

            loss_sum += outputs.loss.item()
            step += 1

    avg_loss = loss_sum / step
    ppl = float(np.exp(avg_loss))
    logger.info(f"Validation Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")

    if out:
        return avg_loss, ppl
    else:
        return avg_loss, ppl


def test(model, TestDataLoader, args, logger, out=False):
    """
    功能: 评估模型在测试集上的性能
    输入:
        - model: InstructTime模型实例
        - TestDataLoader: 测试数据加载器
        - args: 运行参数
        - logger: 日志记录器
        - out: 布尔值，是否返回原始预测结果和标签
    输出:
        - 如果out=False: 返回模型在所有数据集上的综合评估分数(浮点数)
        - 如果out=True: 返回原始预测结果和标签列表(用于详细分析)
    过程:
        1. 将模型设置为评估模式
        2. 不计算梯度，进行前向推理
        3. 解码模型输出，提取各种数据类型的预测结果
        4. 计算各数据类型的评估指标并返回
    """
    model.eval()

    with torch.no_grad():
        pred_ids, pred_eeg, pred_har, pred_fd, pred_rwc = [], [], [], [], []
        labels, labels_eeg, labels_har, labels_fd, labels_rwc = [], [], [], [], []

        all_extracted_info = []
        all_sig_labels = []
        if out:
            print_labels = []
            print_preds = []
        for data in tqdm(TestDataLoader, desc="Eval", ncols=120):
            input_ids = data["input_ids"].to(args.device)
            bt_labels = data["labels"]
            
            outputs = model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                do_sample=False,
                max_new_tokens=args.per_max_token,
            )
            
            mask = outputs >= tokenizer.text_vocab_size
            outputs[mask] = tokenizer.pad_token_id
            outputs = outputs[:, args.encoder_max_length:]
            decoded_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            all_extracted_info.extend([extract_all_information(dt) for dt in decoded_texts])
            all_sig_labels.extend([extract_all_information(label) for label in bt_labels])
            if out:
                print_labels.extend(bt_labels)
                print_preds.extend(decoded_texts)
        
        for decoded_info, sig_label_info in zip(all_extracted_info, all_sig_labels):
            diagnosis_text, stage_text, har_text, fd_text, rwc_text = decoded_info
            diagnosis_label, stage_label, har_label, fd_label, rwc_label = sig_label_info

            if diagnosis_label:
                pred_ids.append(diagnosis_text)
                labels.append(diagnosis_label)

            elif stage_label:
                pred_eeg.append(stage_text)
                labels_eeg.append(stage_label)

            elif har_label:
                pred_har.append(har_text)
                labels_har.append(har_label)
            
            elif fd_label:
                pred_fd.append(fd_text)
                labels_fd.append(fd_label)

            elif rwc_label:
                pred_rwc.append(rwc_text)
                labels_rwc.append(rwc_label)

        res1, res2, res3, res4, res5 = 0, 0, 0, 0, 0
        if args.dataset == 'mix':
            res1, _, _ = metric_ecg(pred_ids, labels, logger)
            res2, _, _ = metric_eeg(pred_eeg, labels_eeg, logger)
            res3, _, _ = metric_har(pred_har, labels_har, logger)
            res4, _, _ = metric_fd(pred_fd, labels_fd, logger)
            res5, _, _ = metric_rwc(pred_rwc, labels_rwc, logger)
        elif args.dataset == 'geo':
            res1, _, _ = metric_ecg(pred_ids, labels, logger)
        elif args.dataset == 'eeg':
            res2, _, _ = metric_eeg(pred_eeg, labels_eeg, logger)
        elif args.dataset == 'fd':
            res3, _, _ = metric_fd(pred_fd, labels_fd, logger)
        elif args.dataset == 'rwc':
            res5, _, _ = metric_rwc(pred_rwc, labels_rwc, logger)
        else:
            res4, _, _ = metric_har(pred_har, labels_har, logger)

    if out:
        return print_preds, print_labels
    else:
        return res1 + res2 + res3 + res4 + res5

def setup_logging(run_path):
    """
    功能: 设置日志记录器
    输入:
        - run_path: 日志文件保存路径
    输出:
        - 配置好的logger对象，用于记录训练过程中的信息
    """
    log_file = os.path.join(run_path, "log.log")


    open(log_file, 'w').close()
    logger = logging.getLogger('training_log')
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=2)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def initialize_model(args, tokenizer, TStokenizers):
    """
    功能: 初始化InstructTime模型
    输入:
        - args: 运行参数
        - tokenizer: MultiTokenizer实例, 用于处理文本和时间序列数据
        - TStokenizers: 时间序列tokenizer列表
    输出:
        - model: 初始化好的InstructTime模型实例
        - sub_path: 模型保存子路径
    过程:
        1. 加载GPT2配置
        2. 初始化InstructTime模型
        3. 加载预训练的GPT2权重
        4. 调整token嵌入层大小
        5. 创建新的输出嵌入层
    """
    config = GPT2Config.from_pretrained(local_model_path)
    model = InstructTime(config, TStokenizers, text_embedding=len(tokenizer.textTokenizer)).to(args.device)

    pretrained_gpt2_model = GPT2LMHeadModel.from_pretrained(local_model_path)
    model.load_state_dict(pretrained_gpt2_model.state_dict(), strict=False)

    # ① 由于新增 <BET>, <EET> 两个文本 token，先把 wte (50259→50261)
    #    也就是 nn.Embedding(vocab_txt, hidden) 扩容。
    #    新行随机 ~ N(0, config.initializer_range²)；lm_head 会一起扩且仍与 wte 绑权重。
    model.resize_token_embeddings(len(tokenizer.textTokenizer))

    # ② 保留“扩好后仍 tied 的 lm_head”权重视图，稍后拷贝到自定义更大 lm_head。
    current_output = model.get_output_embeddings()   # shape (50261, hidden)

    # ③ 新建一个更大的的 lm_head：行 = 文本 50261 + ΣK_i(codebooks)，列 = hidden
    new_output = nn.Linear(config.n_embd,
                        tokenizer.vocabSize_all(),   # 50261 + ΣK_i
                        bias=False).to(args.device)

    # ④ 继承文本部分权重；codebook 行保持随机初始化
    new_output.weight.data[:len(tokenizer.textTokenizer)] = current_output.weight.data

    # ⑤ 替换输出投影，解除 wte 与 lm_head 的 weight-tying
    model.set_output_embeddings(new_output)
    #    - 输入端 wte 只覆盖 0-50260 的文本 ID
    #    - 输出端可预测文本 ID + 各 codebook token
    #    - codebook token 的输入嵌入由 TSEmbedding 负责
    
    sub_path = "no_frozen"
    
    return model, sub_path

def train_model(model, args, TrainDataLoader, ValidDataLoader, optimizer, scheduler, scaler, logger, run_path):
    """
    功能: 训练InstructTime模型
    输入:
        - model: InstructTime模型实例
        - args: 运行参数
        - TrainDataLoader: 训练数据加载器
        - ValidDataLoader: 验证数据加载器
        - optimizer: 优化器
        - scheduler: 学习率调度器
        - scaler: 梯度缩放器(用于混合精度训练)
        - logger: 日志记录器
        - run_path: 结果保存路径
    输出:
        - 无直接返回值, 但会将最佳模型保存到run_path/best_model路径
    过程:
        1. 训练每个epoch:
            a. 在训练集上更新模型参数
            b. 记录训练损失
        2. 在验证集上评估模型
        3. 保存性能最佳的模型
    """
    best_ppl = float("inf")
    best = 0.0
    patience = 3
    patience_cnt = 0
        
    for epoch in range(args.epochs): 
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(TrainDataLoader, desc=f"GPT Epoch {epoch+1}", ncols=120)
        
        model.train()
        for data in tqdm_iter:

            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            label_ids = data["label_ids"].to(args.device)
            
            with autocast():
                outputs = model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=label_ids
                            )
            
            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            loss_value = outputs.loss.cpu().item()
            train_losses += loss_value
            step += 1
            tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})

        train_loss = train_losses / step
        logger.info(f"Epoch {epoch+1}\nLoss: {train_loss:.4f}")

        if args.adapt: # SFT
            res = validate_sft(model, ValidDataLoader, args, logger, out=False)
            print(res)
            if res > best:
                MODEL_STORED_PATH = run_path + "/best_model"
                best = res
                model.save_pretrained(MODEL_STORED_PATH)
        else: # AR
            val_loss, val_ppl = validate_ar(model, ValidDataLoader, args, logger)

            print(f"Epoch {epoch+1:02d} | "
                  f"TrainLoss {train_loss:.4f} | "
                  f"ValLoss {val_loss:.4f} | "
                  f"ValPPL {val_ppl:.2f}")

            if args.wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                })

            if val_ppl < best_ppl:
                best_ppl = val_ppl
                patience_cnt = 0
                MODEL_STORED_PATH = run_path + "/best_model"
                model.save_pretrained(MODEL_STORED_PATH)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logger.info("Early stopping triggered")
                    break

if __name__ == "__main__":
    args = get_hyperparams()
    seed_everything(args.seed)

    if args.wandb:
        wandb.init(
            project  = "Instructime Replicate",
            name     = f"{args.model}_lr{args.lr}",  # run name
            group    = "LLM_AR",           # 同一实验组
            config   = vars(args),                  # 超参入库
        )

    # 构建训练、验证和测试数据集路径
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
        train_files = os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "train.pkl")
        val_files = os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "val.pkl")
        test_files = os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "test.pkl")

    # 加载时间序列tokenizer
    ts_tokenizer = load_TStokenizer(vqvae_path, data_shape=(48, 34), device="cpu")
    tokenizer = MultiTokenizer([ts_tokenizer])

    # 构建数据集
    TrainDataset = MultiDataset(train_files, tokenizer, mode="train",
                                encoder_max_length=args.encoder_max_length,
                                loss_type="ar")
    ValidDataset = MultiDataset(val_files, tokenizer, mode="train",
                                encoder_max_length=args.encoder_max_length,
                                loss_type="ar")
    TestDataset = MultiDataset(test_files, tokenizer, mode="test",
                               encoder_max_length=args.encoder_max_length)

    # smoke test 时仅取前30条样本
    if args.smoke_test:
        TrainDataset.samples = TrainDataset.samples[:30]
        ValidDataset.samples = ValidDataset.samples[:30]
        TestDataset.samples = TestDataset.samples[:30]

    # 数据加载器
    TrainDataLoader = DataLoader(
        TrainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    ValidDataLoader = DataLoader(
        ValidDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    TestDataLoader = DataLoader(
        TestDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    num = 1
    for run in range(num):
        model, sub_path = initialize_model(args, tokenizer, [ts_tokenizer])
        model_subpath = os.path.join(args.model_path, sub_path)
        print(args.model_path, model_subpath)

        os.makedirs(model_subpath, exist_ok=True)
        run_path = os.path.join(model_subpath, f"run_{run}")
        os.makedirs(run_path, exist_ok=True)
        logger = setup_logging(run_path)
        
        if args.adapt:
            best_model_path = os.path.join(run_path, 'best_model')
            model_state_dict = torch.load(os.path.join(args.load_model_path, 'pytorch_model.bin'), map_location=args.device)
            model.load_state_dict(model_state_dict, strict=False)

        for param in model.parameters():
            param.requires_grad = True
            
        param_dict = [{"params": model.parameters(), "lr": args.lr}]
        optimizer = torch.optim.Adam(param_dict, weight_decay=1e-5)
        scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.epochs * len(TrainDataLoader) * args.warm_up_ratio, num_training_steps=args.epochs * len(TrainDataLoader)
        )
        scaler = GradScaler()

        logger.info(f"Begin training for run {run}")
        train_model(model, args, TrainDataLoader, ValidDataLoader,
                    optimizer, scheduler, scaler, logger, run_path)

        # if args.adapt:
        #     model, _ = initialize_model(args, tokenizer)
        #     best_model_path = os.path.join(run_path, 'best_model')
        #     model_state_dict = torch.load(
        #         os.path.join(best_model_path, 'pytorch_model.bin'),
        #         map_location=args.device)
        #     model.load_state_dict(model_state_dict)

        #     logger.info(f"Test best model for run {run}")
        #     print_preds, print_labels = test(model, TestDataLoader, args, logger, out=True)

        #     save_path = os.path.join(run_path, 'output.txt')
        #     with open(save_path, 'w', encoding='utf-8') as file:
        #         for i in range(len(print_labels)):
        #             j = i * args.num_return_sequences
        #             for k in range(args.num_return_sequences):
        #                 file.write(f"Generated Text: {print_preds[j + k]}\n")
        #             file.write(f"Actual Label: {print_labels[i]}\n\n")

        logger.handlers.clear()

    if args.wandb:
        wandb.finish()

# # Universal模型(AR)的smoke test
# python run_truth_loss.py --device "cuda:0" --dataset "mix" --batch_size 4 --lr 1e-5 --epochs 1 --adapt False --smoke_test True

# # Adapt模型(SFT)的smoke test
# python run_truth_loss.py --device "cuda:0" --dataset "ecg" --batch_size 4 --lr 1e-5 --epochs 1 --adapt True --load_model_path "./gptmodel" --smoke_test True
