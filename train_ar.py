import os
import logging
import random
from logging.handlers import RotatingFileHandler
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import transformers
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
import wandb
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from multimodel import InstructTime, InstructTimeLlama, MultiTokenizer
from datamodules import build_ar_dataloaders
from utils import load_TStokenizer

vqvae_path = "TStokenizer/Vq_weight"
DATA_ROOT = "ts_tokenized_datasets"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def setup_logging(run_path: str) -> logging.Logger:
    log_file = os.path.join(run_path, "log.log")
    open(log_file, "w").close()
    logger = logging.getLogger("training_log")
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5, backupCount=2)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def initialize_model(device: str, tokenizer: MultiTokenizer, ts_tokenizers, base_model: str, pretrained: str):
    """根据 base_model 构建对应的 InstructTime* 模型"""
    if base_model.lower() == "llama3":
        # ---------- Llama3 版本 ----------
        config = LlamaConfig.from_pretrained(pretrained)
        model = InstructTimeLlama(config, ts_tokenizers, text_embedding=len(tokenizer.textTokenizer)).to(device)
        pretrained_llama_model = LlamaForCausalLM.from_pretrained(pretrained)
        model.load_state_dict(pretrained_llama_model.state_dict(), strict=False)

        model.resize_token_embeddings(len(tokenizer.textTokenizer))
        current_output = model.get_output_embeddings()
        new_output = nn.Linear(config.hidden_size, tokenizer.vocabSize_all(), bias=False).to(device)
        new_output.weight.data[: len(tokenizer.textTokenizer)] = current_output.weight.data
        model.set_output_embeddings(new_output)
        model.config.vocab_size = tokenizer.vocabSize_all()
        
        # 添加LoRA配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        sub_path = "lora"
    else:
        # ---------- GPT-2 版本 ----------
        config = GPT2Config.from_pretrained(pretrained)
        model = InstructTime(config, ts_tokenizers, text_embedding=len(tokenizer.textTokenizer)).to(device)
        pretrained_gpt2_model = GPT2LMHeadModel.from_pretrained(pretrained)
        model.load_state_dict(pretrained_gpt2_model.state_dict(), strict=False)

        model.resize_token_embeddings(len(tokenizer.textTokenizer))
        current_output = model.get_output_embeddings()
        new_output = nn.Linear(config.n_embd, tokenizer.vocabSize_all(), bias=False).to(device)
        new_output.weight.data[: len(tokenizer.textTokenizer)] = current_output.weight.data
        model.set_output_embeddings(new_output)
        model.config.vocab_size = tokenizer.vocabSize_all()
        sub_path = "no_frozen"

    return model, sub_path


def validate_ar(model, loader, device, logger):
    """
    在validaton set上计算PPL与Loss
    PPL 是在整个序列上计算的：
    输入序列：[问题部分] + [答案部分] + [EOS]
    标签序列：与输入序列相同
    损失计算：模型对整个序列进行 next-token prediction, 使用标准的 shift-1 交叉熵损失
    PPL 计算：基于整个序列的平均损失
    """
    model.eval()
    loss_sum = 0.0
    step = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Val", ncols=120):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            label_ids = data["label_ids"].to(device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
            loss_sum += outputs.loss.item()
            step += 1
    avg_loss = loss_sum / step
    ppl = float(np.exp(avg_loss))
    print(f"Validation Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")
    return avg_loss, ppl


def train_model(model, args, train_loader, valid_loader, optimizer, scheduler, scaler, logger, run_path):
    best_ppl = float("inf")
    patience = 3
    patience_cnt = 0
    for epoch in range(args.epochs):
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=120)
        model.train()
        for data in tqdm_iter:
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            label_ids = data["label_ids"].to(args.device)
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
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
        print(f"Epoch {epoch+1}\nLoss: {train_loss:.4f}")
        val_loss, val_ppl = validate_ar(model, valid_loader, args.device, logger)
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
            if isinstance(model, PeftModel):
                # 保存基座 + 适配器
                model.save_pretrained(os.path.join(run_path, "best_model"),
                                    safe_serialization=True,
                                    save_adapter=True)           # adapter
                model.base_model.save_pretrained(os.path.join(run_path, "best_model"),
                                                safe_serialization=True)   # base
            else:
                model.save_pretrained(os.path.join(run_path, "best_model"), safe_serialization=True)

        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early stopping triggered")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="./production_model", help="path to save AR model")
    parser.add_argument("--base_model", type=str, default="gpt2", choices=["gpt2", "llama3"], help="base model")
    parser.add_argument("--pretrained_model", type=str, default="gpt2", help="HF model name or local path")
    parser.add_argument("--dataset", type=str, default="mix", choices=["mix", "ihm", "pheno"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_max_length", type=int, default=230)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warm_up_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.wandb:
        wandb.init(project="Instructime Replicate", group="LLM_AR", config=vars(args))

    if args.dataset == "mix":
        train_files = [
            os.path.join(DATA_ROOT, "ihm", "tokenized_v1_patch64", "train.pkl"),
            os.path.join(DATA_ROOT, "pheno", "tokenized_v1_patch64", "train.pkl"),
        ]
        val_files = [
            os.path.join(DATA_ROOT, "ihm", "tokenized_v1_patch64", "val.pkl"),
            os.path.join(DATA_ROOT, "pheno", "tokenized_v1_patch64", "val.pkl"),
        ]
    else:
        train_files = os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "train.pkl")
        val_files = os.path.join(DATA_ROOT, args.dataset, "tokenized_v1_patch64", "val.pkl")

    ts_tokenizer = load_TStokenizer(vqvae_path, data_shape=(48, 34), device="cpu")
    tokenizer = MultiTokenizer([ts_tokenizer], base_model=args.base_model)

    train_loader, val_loader = build_ar_dataloaders(train_files, val_files, tokenizer, args.batch_size, args.encoder_max_length)

    if args.smoke_test:
        train_loader.dataset.samples = train_loader.dataset.samples[:30]
        val_loader.dataset.samples = val_loader.dataset.samples[:30]

    model, sub_path = initialize_model(args.device, tokenizer, [ts_tokenizer], args.base_model, args.pretrained_model)
    model_subpath = os.path.join(args.model_path, sub_path)
    os.makedirs(model_subpath, exist_ok=True)
    run_path = os.path.join(model_subpath, "run_0")
    os.makedirs(run_path, exist_ok=True)
    logger = setup_logging(run_path)

    # 设置参数训练状态
    if args.base_model.lower() == "llama3":
        # 对于Llama3 LoRA模型，只训练可训练的参数
        trainable_params = 0
        all_params = 0
        for param in model.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"可训练参数: {trainable_params:,} / 总参数: {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
        optimizer = torch.optim.Adam([{"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr}], weight_decay=1e-5)
    else:
        # 对于GPT-2模型，训练所有参数
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}], weight_decay=1e-5)

    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.epochs * len(train_loader) * args.warm_up_ratio,
        num_training_steps=args.epochs * len(train_loader),
    )
    scaler = GradScaler()

    # 打印训练样本数量
    train_samples_count = len(train_loader.dataset)
    val_samples_count = len(val_loader.dataset)
    print(train_loader.dataset.samples[0])
    print(f"训练样本总数: {train_samples_count}")
    print(f"验证样本总数: {val_samples_count}")

    print("Begin training")
    train_model(model, args, train_loader, val_loader, optimizer, scheduler, scaler, logger, run_path)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


# 使用 GPT-2 作为基座模型进行 AR 训练
# python train_ar.py \
#     --seed 42 \
#     --base_model gpt2 \
#     --pretrained_model gpt2 \
#     --dataset mix \
#     --batch_size 32 \
#     --encoder_max_length 1024 \
#     --lr 1e-5 \
#     --warm_up_ratio 0.05 \
#     --epochs 50 \
#     --model_path ./production_model_gpt2 \
#     --device cuda:0 \
#     --wandb

# 使用 Llama3 作为基座模型进行 AR 训练 (LoRA微调)
# python train_ar.py \
#     --seed 42 \
#     --base_model llama3 \
#     --pretrained_model meta-llama/Meta-Llama-3-8B \
#     --dataset mix \
#     --batch_size 16 \
#     --encoder_max_length 1024 \
#     --lr 5e-6 \
#     --warm_up_ratio 0.05 \
#     --epochs 30 \
#     --model_path ./production_model_llama3 \
#     --device cuda:0 \
#     --wandb

# 快速测试 (smoke test) - GPT-2
# python train_ar.py \
#     --base_model gpt2 \
#     --pretrained_model gpt2 \
#     --dataset ihm \
#     --batch_size 8 \
#     --epochs 2 \
#     --smoke_test

# 快速测试 (smoke test) - Llama3 (LoRA微调)
# python train_ar.py \
#     --base_model llama3 \
#     --pretrained_model meta-llama/Meta-Llama-3-8B \
#     --dataset ihm \
#     --batch_size 4 \
#     --epochs 2 \
#     --smoke_test
