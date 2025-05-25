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
from transformers import GPT2Config, GPT2LMHeadModel
import wandb

from multimodel import InstructTime, MultiTokenizer
from datamodules import build_sft_dataloaders
from utils import load_TStokenizer
from multidataset import PHENO_LABELS
from metrics import compute_metrics

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


def initialize_model(device: str, tokenizer: MultiTokenizer, ts_tokenizers, weight_path: str):
    config = GPT2Config.from_pretrained(weight_path)
    model = InstructTime(config, ts_tokenizers, text_embedding=len(tokenizer.textTokenizer)).to(device)
    pretrained_gpt2_model = GPT2LMHeadModel.from_pretrained(weight_path)
    model.load_state_dict(pretrained_gpt2_model.state_dict(), strict=False)

    model.resize_token_embeddings(len(tokenizer.textTokenizer))
    current_output = model.get_output_embeddings()
    new_output = nn.Linear(config.n_embd, tokenizer.vocabSize_all(), bias=False).to(device)
    new_output.weight.data[: len(tokenizer.textTokenizer)] = current_output.weight.data
    model.set_output_embeddings(new_output)
    model.config.vocab_size = tokenizer.vocabSize_all()

    sub_path = "no_frozen"
    return model, sub_path


IHM_LABEL_MAP = {
    "the patient will survive": 0,
    "the patient will die": 1,
}
PHENO_LABEL_MAP = {l.lower(): i for i, l in enumerate(PHENO_LABELS)}


def validate_sft(model, loader, device, tokenizer, dataset, logger):
    """Generate answers and compute F1/ACC metrics."""
    model.eval()
    pred_ihm, gold_ihm = [], []
    pred_pheno, gold_pheno = [], []
    with torch.no_grad():
        for data in tqdm(loader, desc="Val", ncols=120):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            labels = data["label"]

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                num_beams=1,
            )

            for g, l in zip(gen_ids, labels):
                text = tokenizer.decode(g.tolist(), skip_special_tokens=True)
                if l.dim() == 0 or l.numel() == 1:
                    pred_ihm.append(text)
                    gold_ihm.append(int(l.item()))
                else:
                    pred_pheno.append(text)
                    gold_pheno.append(l.numpy().tolist())

    results = {}
    if pred_ihm:
        results["ihm"] = compute_metrics(pred_ihm, gold_ihm, "ihm", IHM_LABEL_MAP)
        logger.info(
            f"IHM Val ACC: {results['ihm']['acc']:.4f} | F1: {results['ihm']['f1']:.4f}"
        )
    if pred_pheno:
        results["pheno"] = compute_metrics(pred_pheno, gold_pheno, "pheno", PHENO_LABEL_MAP)
        logger.info(
            f"Pheno Val ACC: {results['pheno']['acc']:.4f} | F1: {results['pheno']['f1']:.4f}"
        )
    return results


def train_model(model, args, train_loader, valid_loader, optimizer, scheduler, scaler, logger, run_path, tokenizer):
    best_f1 = -1.0
    patience = 3
    patience_cnt = 0
    for epoch in range(args.epochs):
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(train_loader, desc=f"GPT Epoch {epoch+1}", ncols=120)
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
        logger.info(f"Epoch {epoch+1}\nLoss: {train_loss:.4f}")
        metrics = validate_sft(model, valid_loader, args.device, tokenizer, args.dataset, logger)

        if args.dataset == "ihm":
            val_f1 = metrics["ihm"]["f1"]
        elif args.dataset == "pheno":
            val_f1 = metrics["pheno"]["f1"]
        else:
            val_f1 = (metrics["ihm"]["f1"] + metrics["pheno"]["f1"]) / 2

        if args.wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            model.save_pretrained(os.path.join(run_path, "best_model"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("Early stopping triggered")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="./gptmodel")
    parser.add_argument("--init_model_path", type=str, required=True, help="path to pretrained AR model")
    parser.add_argument("--dataset", type=str, default="mix", choices=["mix", "ihm", "pheno"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_max_length", type=int, default=230)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warm_up_ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.wandb:
        wandb.init(project="Instructime Replicate", group="LLM_SFT", config=vars(args))

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

    ts_tokenizer = load_TStokenizer(vqvae_path, data_shape=(48, 34), device="cpu")
    tokenizer = MultiTokenizer([ts_tokenizer])

    train_loader, val_loader, _ = build_sft_dataloaders(
        train_files,
        val_files,
        test_files,
        tokenizer,
        args.batch_size,
        args.encoder_max_length,
    )

    if args.smoke_test:
        train_loader.dataset.samples = train_loader.dataset.samples[:30]
        val_loader.dataset.samples = val_loader.dataset.samples[:30]

    model, sub_path = initialize_model(args.device, tokenizer, [ts_tokenizer], args.init_model_path)
    model_subpath = os.path.join(args.model_path, sub_path)
    os.makedirs(model_subpath, exist_ok=True)
    run_path = os.path.join(model_subpath, "run_0")
    os.makedirs(run_path, exist_ok=True)
    logger = setup_logging(run_path)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.lr}], weight_decay=1e-5)
    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.epochs * len(train_loader) * args.warm_up_ratio,
        num_training_steps=args.epochs * len(train_loader),
    )
    scaler = GradScaler()

    logger.info("Begin training")
    train_model(model, args, train_loader, val_loader, optimizer, scheduler, scaler, logger, run_path, tokenizer)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
