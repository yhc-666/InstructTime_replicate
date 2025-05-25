"""Datasets for autoregressive (AR) and supervised fine-tuning (SFT) training.

Each dataset supports ``train`` and ``test`` modes.  ``train`` mode is used for
both training and validation.  ``test`` mode is used for evaluation.

AR test samples return ``label_ids`` for computing perplexity while SFT test
samples return the raw numerical ``label`` for future metric computation.
"""

from __future__ import annotations

import pickle
from typing import List, Sequence, Union

import torch
from torch.utils.data import Dataset

from multimodel import MultiTokenizer

# ---------------------------------------------------------------------------
# Phenotype label names and prompt list
PHENO_LABELS = [
    "Acute and unspecified renal failure",
    "Acute cerebrovascular disease",
    "Acute myocardial infarction",
    "Cardiac dysrhythmias",
    "Chronic kidney disease",
    "Chronic obstructive pulmonary disease and bronchiectasis",
    "Complications of surgical procedures or medical care",
    "Conduction disorders",
    "Congestive heart failure; nonhypertensive",
    "Coronary atherosclerosis and other heart disease",
    "Diabetes mellitus with complications",
    "Diabetes mellitus without complication",
    "Disorders of lipid metabolism",
    "Essential hypertension",
    "Fluid and electrolyte disorders",
    "Gastrointestinal hemorrhage",
    "Hypertension with complications and secondary hypertension",
    "Other liver diseases",
    "Other lower respiratory disease",
    "Other upper respiratory disease",
    "Pleurisy; pneumothorax; pulmonary collapse",
    "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
    "Respiratory failure; insufficiency; arrest (adult)",
    "Septicemia (except in labor)",
    "Shock",
]

PHENO_LIST_TEXT = (
    "  Acute and unspecified renal failure,\n"
    "  Acute cerebrovascular disease,\n"
    "  Acute myocardial infarction,\n"
    "  Cardiac dysrhythmias,\n"
    "  Chronic kidney disease,\n"
    "  Chronic obstructive pulmonary disease and bronchiectasis,\n"
    "  Complications of surgical procedures or medical care,\n"
    "  Conduction disorders,\n"
    "  Congestive heart failure; nonhypertensive,\n"
    "  Coronary atherosclerosis and other heart disease,\n"
    "  Diabetes mellitus with complications,\n"
    "  Diabetes mellitus without complication,\n"
    "  Disorders of lipid metabolism,\n"
    "  Essential hypertension,\n"
    "  Fluid and electrolyte disorders,\n"
    "  Gastrointestinal hemorrhage,\n"
    "  Hypertension with complications and secondary hypertension,\n"
    "  Other liver diseases,\n"
    "  Other lower respiratory disease,\n"
    "  Other upper respiratory disease,\n"
    "  Pleurisy; pneumothorax; pulmonary collapse,\n"
    "  Pneumonia (except that caused by tuberculosis or sexually transmitted disease),\n"
    "  Respiratory failure; insufficiency; arrest (adult),\n"
    "  Septicemia (except in labor),\n"
    "  Shock."
)


class _BaseDataset(Dataset):
    """Base dataset shared by AR and SFT training."""

    def __init__(
        self,
        pkl_files: Union[str, Sequence[str]],
        tokenizer: MultiTokenizer,
        mode: str,
        encoder_max_length: int = 512,
        model_id: int = 0,
    ) -> None:
        assert mode in {"train", "test"}
        super().__init__()

        if isinstance(pkl_files, str):
            pkl_files = [pkl_files]

        self.samples = []
        for path in pkl_files:
            with open(path, "rb") as f:
                data = pickle.load(f)
            task = "pheno" if "pheno" in path.lower() else "ihm"
            for item in data:
                self.samples.append(
                    {
                        "ts_ids": item["ts_ids"],
                        "notes": item["notes"][-2:-1],
                        "label": item["label"],
                        "task": task,
                    }
                )

        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = encoder_max_length
        self.model_id = model_id

        # constant tokens
        self.bet_id = self.tokenizer.encode("<BET>")[0]
        self.eet_id = self.tokenizer.encode("<EET>")[0]
        self.offset = self.tokenizer.offsets[model_id]

    # ------------------------------------------------------------------
    # helpers
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def _build_prompt(self, sample: dict) -> tuple[List[int], str]:
        """Construct base input ids and label string from a raw sample."""

        notes_text = "\n".join(sample["notes"])

        if sample["task"] == "ihm":
            instruction = (
                "Given the 48-hour ICU multivariate vital-sign sequence and doctors' notes below, "
                "predict whether the patient will die in hospital.\n"
                "The possible outcomes are: the patient will survive; the patient will die."
            )
            label_text = (
                "The patient will die." if int(sample["label"]) == 1 else "The patient will survive."
            )
        else:
            instruction = (
                "Given the 24-hour ICU multivariate vital-sign sequence and doctors' notes below, "
                "select all phenotypes that apply to this ICU stay and briefly state them.\n"
                "The 25 possible phenotypes include:\n" + PHENO_LIST_TEXT
            )
            pos = [i for i, v in enumerate(sample["label"]) if int(v) == 1]
            labels = [PHENO_LABELS[i] for i in pos]
            label_text = "The patient presents " + " and ".join(labels)

        prompt_text = (
            f"{instruction}\n\nDoctors' notes:\n{notes_text}\n\nVital-sign time-series:\n"
        )
        prompt_ids = self.tokenizer.encode(prompt_text)

        ts_ids = [tid + self.offset for tid in sample["ts_ids"]]
        # text appended after <EET> token, leading period and answer indicator
        after_ids = self.tokenizer.encode(".\n\nAnswer Output:\n")
        base_ids = prompt_ids + [self.bet_id] + ts_ids + [self.eet_id] + after_ids
        return base_ids, label_text

    def _padding(self, ids: List[int], masks: List[int]):
        """Pad sequences to ``self.max_length``."""
        ids = ids[: self.max_length]
        masks = masks[: self.max_length]

        pad_len = self.max_length - len(ids)
        if self.mode == "train":
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
            masks = masks + [0] * pad_len
        else:
            ids = [self.tokenizer.pad_token_id] * pad_len + ids
            masks = [0] * pad_len + masks
        return ids, masks


class ARDataset(_BaseDataset):
    """Dataset for autoregressive (universal) training."""

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.samples[idx]
        base_ids, label_text = self._build_prompt(sample)

        label_ids = self.tokenizer.encode(" " + label_text)
        eos_id = self.tokenizer.eos_token_id

        # ARDataset 只支持训练/验证时采用相同的数据格式
        input_ids = base_ids + label_ids + [eos_id]
        labels = input_ids.copy()

        attn_masks = [1] * len(input_ids)
        input_ids, attn_masks = self._padding(input_ids, attn_masks)
        labels, _ = self._padding(labels, attn_masks)

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attn_masks": torch.FloatTensor(attn_masks),
            "label_ids": torch.LongTensor(labels),
        }


class SFTDataset(_BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __getitem__(self, idx: int):  # type: ignore[override]
        sample = self.samples[idx]
        base_ids, label_text = self._build_prompt(sample)

        label_ids = self.tokenizer.encode(" " + label_text)
        eos_id = self.tokenizer.eos_token_id

        if self.mode == "train": # train模式
            input_ids = base_ids + label_ids + [eos_id]
            labels = [-100] * len(base_ids) + label_ids + [eos_id]
            attn_masks = [1] * len(input_ids)
            input_ids, attn_masks = self._padding(input_ids, attn_masks)
            labels, _ = self._padding(labels, attn_masks)
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "label_ids": torch.LongTensor(labels),
            }

        if self.mode == "test":
            # validation/test模式：只提供问题部分，让模型生成答案
            input_ids = base_ids  # 不添加EOS，让模型从"Answer Output:\n"后开始生成
            attn_masks = [1] * len(input_ids)
            input_ids, attn_masks = self._padding(input_ids, attn_masks)

            label_tensor = (
                torch.tensor(sample["label"], dtype=torch.long)
                if isinstance(sample["label"], list)
                else torch.tensor(int(sample["label"]), dtype=torch.long)
            )

            return {
                "input_ids": torch.LongTensor(input_ids), # [问题部分], 用于输入模型生成回复
                "attn_masks": torch.FloatTensor(attn_masks),
                "label": label_tensor, # 原始标签, 用于结合模型生成的回复计算AUROC/AUPRC/F1指标
            }

