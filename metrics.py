"""
作用: 提供多种时间序列领域的评估指标计算函数，用于模型性能评估
输入: 
    - 模型预测结果(文本形式)
    - 真实标签
    - logger对象(用于记录)
输出: 
    - 评估分数(根据领域不同有不同指标)
    - 详细的评估报告(记录在日志中)
示例用法:
    # 评估心电图(ECG)数据的分类结果
    score, report, matrix = metric_ecg(predictions, ground_truth, logger)
    
    # 评估脑电图(EEG)数据的分类结果
    score, report, matrix = metric_eeg(predictions, ground_truth, logger)
    
    # 评估人体活动识别(HAR)结果
    score, report, matrix = metric_har(predictions, ground_truth, logger)
"""

import numpy as np
import pandas as pd
#from challeng_score import evaluate_model
from sklearn.metrics import f1_score, hamming_loss
from typing import List, Union, Dict

# def get_dict(Path):
#     mapping_file = Path
#     mapping_data = pd.read_csv(mapping_file)

#     annotation_to_condition = {}
#     for index, row in mapping_data.iterrows():
#         annotation_to_condition[row['Full Name']] = index
    
#     return annotation_to_condition

# def encode_labels(label_dict, label_str, delimiter=','):
#     labels = label_str.split(delimiter)
#     encoded = [0] * len(label_dict)
#     for label in labels:
#         label = label.strip()
#         if label not in label_dict:
#             continue
#         encoded[label_dict[label]] = 1

#     return encoded

# def metric_ecg(preds, labels, logger, delimiter=','):
#     diction = get_dict(Path='essy.csv')

#     print(preds[0])
#     print(labels[0])

#     encoded_preds = np.array([encode_labels(diction, p, delimiter) for p in preds])
#     encoded_labels = np.array([encode_labels(diction, l, delimiter) for l in labels])
    
#     zero_preds = []
#     zero_labels = []
#     count = 0
#     for i, encoded_pred in enumerate(encoded_preds):
#         if np.all(encoded_pred == 0):
#             zero_preds.append(preds[i])
#             zero_labels.append(labels[i])
#             count += 1
    
#     print(count / len(preds))

#     print(encoded_preds[0])
#     print(encoded_labels[0])

#     hit1 = np.mean(np.all(encoded_preds == encoded_labels, axis=1))
#     total_f1 = f1_score(encoded_labels, encoded_preds, average='samples', zero_division=0)
#     hloss = hamming_loss(encoded_labels, encoded_preds)
#     _, score = evaluate_model(encoded_labels, encoded_preds)
    
#     logger.info(
#         "Evaluation result:\naccuracy: {}\nTotal F1: {}\nHmloss: {}\nScore: {}\n".format(
#             hit1,
#             total_f1,
#             hloss,
#             score
#         )
#     )

#     print(
#         "accuracy: {}\nTotal F1: {}\nHmloss: {}\nScore: {}\n".format(
#             hit1,
#             total_f1,
#             hloss,
#             score
#         )
#     )
#     return hit1, zero_preds, zero_labels

# def metric_eeg(preds_eeg, labels_eeg, logger):
#     sleep_stages = {
#         'waking up': 0,
#         'rapid eye movement sleep': 1,
#         'sleep stage one': 2,
#         'sleep stage two': 3,
#         'sleep stage three': 4,
#         'sleep stage four': 5,
#         'period of movement': 6,
#         'unidentified stage': 7
#     }

#     print(preds_eeg[0])
#     print(labels_eeg[0])

#     preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds_eeg])
#     labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels_eeg])
    
#     zero_preds = []
#     zero_labels = []
#     count = 0
#     for i, encoded_pred in enumerate(preds_mapped):
#         if encoded_pred == -1:
#             zero_preds.append(preds_eeg[i])
#             zero_labels.append(labels_eeg[i])
#             count += 1
    
#     print(count / len(preds_eeg))

#     print(preds_mapped[0])
#     print(labels_mapped[0])

#     hit2 = np.mean(preds_mapped == labels_mapped)
#     sleep_f1 = f1_score(labels_mapped, preds_mapped, average='macro', zero_division=0)
    
#     logger.info(
#         "Sleep Evaluation result:\naccuracy: {}\nTotal F1 sleep: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     print(
#         "accuracy: {}\nTotal F1 sleep: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     return hit2, zero_preds, zero_labels

# def metric_har(preds, labels, logger):
#     sleep_stages = {
#         'walking': 0,
#         'ascending stairs': 1,
#         'descending stairs': 2,
#         'sitting': 3,
#         'standing': 4,
#         'lying down': 5
#     }

#     print(preds[0])
#     print(labels[0])

#     preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds])
#     labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels])
    
#     zero_preds = []
#     zero_labels = []
#     count = 0
#     for i, encoded_pred in enumerate(preds_mapped):
#         if encoded_pred == -1:
#             zero_preds.append(preds[i])
#             zero_labels.append(labels[i])
#             count += 1
    
#     print(count / len(preds))

#     print(preds_mapped[0])
#     print(labels_mapped[0])

#     hit2 = np.mean(preds_mapped == labels_mapped)
#     sleep_f1 = f1_score(labels_mapped, preds_mapped, average='macro', zero_division=0)
    
#     logger.info(
#         "HAR Evaluation result:\naccuracy: {}\nTotal F1 HAR: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     print(
#         "accuracy: {}\nTotal F1 HAR: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     return hit2, zero_preds, zero_labels

# def metric_fd(preds, labels, logger):
#     sleep_stages = {
#         'not damaged': 0,
#         'inner damaged': 1,
#         'outer damaged': 2,
#     }

#     print(preds[0])
#     print(labels[0])

#     preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds])
#     labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels])
    
#     zero_preds = []
#     zero_labels = []
#     count = 0
#     for i, encoded_pred in enumerate(preds_mapped):
#         if encoded_pred == -1:
#             zero_preds.append(preds[i])
#             zero_labels.append(labels[i])
#             count += 1
    
#     print(count / len(preds))
            
#     print(preds_mapped[0])
#     print(labels_mapped[0])

#     hit2 = np.mean(preds_mapped == labels_mapped)
#     sleep_f1 = f1_score(labels_mapped, preds_mapped, average='macro', zero_division=0)
    
#     logger.info(
#         "FD Evaluation result:\naccuracy: {}\nTotal F1 FD: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     print(
#         "accuracy: {}\nTotal F1 FD: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     return hit2, zero_preds, zero_labels

# def metric_rwc(preds, labels, logger):
#     sleep_stages = {
#         'the right whale': 0,
#         'unknown creature': 1,
#     }

#     print(preds[0])
#     print(labels[0])

#     preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds])
#     labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels])
    
#     zero_preds = []
#     zero_labels = []
#     count = 0
#     for i, encoded_pred in enumerate(preds_mapped):
#         if encoded_pred == -1:
#             zero_preds.append(preds[i])
#             zero_labels.append(labels[i])
#             count += 1
    
#     print(count / len(preds))
            
#     print(preds_mapped[0])
#     print(labels_mapped[0])

#     valid_indices = preds_mapped != -1
#     valid_preds = preds_mapped[valid_indices]
#     valid_labels = labels_mapped[valid_indices]

#     hit2 = np.mean(preds_mapped == labels_mapped)
#     sleep_f1 = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
    
#     logger.info(
#         "RWC Evaluation result:\naccuracy: {}\nTotal F1 RWC: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     print(
#         "accuracy: {}\nTotal F1 RWC: {}\n".format(
#             hit2,
#             sleep_f1
#         )
#     )

#     return hit2, zero_preds, zero_labels


def compute_metrics(
    pred_texts: List[str],
    gold_labels: Union[np.ndarray, List],
    task: str,
    label_map: Dict[str, int],
) -> Dict[str, float]:
    """Compute F1 score and accuracy for IHM or Pheno tasks.

    Parameters
    ----------
    pred_texts : List[str]
        LLM decode后的句子列表
    gold_labels : Union[np.ndarray, List]
        IHM场景下为`list[int]`, PHENO场景下为多热编码的 ``np.ndarray``
    task : str
        ``ihm`` 或 ``pheno``
    label_map : Dict[str, int]
        文本标签与索引的映射 ``{"atrial fibrillation":0, ...}``

    Returns
    -------
    Dict[str, float]
        ``{"f1": float, "acc": float}``
    """

    if task not in {"ihm", "pheno"}:
        raise ValueError("task must be 'ihm' or 'pheno'")

    gold_array = np.array(gold_labels)
    preds = []

    if task == "ihm":
        for text in pred_texts:
            text_l = text.lower()
            pred = -1
            for k, v in label_map.items():
                if k.lower() in text_l:
                    pred = v
                    break
            preds.append(pred)

        preds_arr = np.array(preds)
        acc = float(np.mean(preds_arr == gold_array))
        valid_mask = preds_arr != -1
        if valid_mask.any():
            f1 = float(
                f1_score(gold_array[valid_mask], preds_arr[valid_mask], average="binary", zero_division=0)
            )
        else:
            f1 = 0.0
        return {"f1": f1, "acc": acc}

    # pheno task (multi-label)
    for text in pred_texts:
        text_l = text.lower()
        vec = np.zeros(len(label_map), dtype=int)
        for k, v in label_map.items():
            if k.lower() in text_l:
                vec[v] = 1
        preds.append(vec)

    preds_arr = np.array(preds)
    acc = float(np.mean(np.all(preds_arr == gold_array, axis=1)))
    f1 = float(f1_score(gold_array, preds_arr, average="samples", zero_division=0))
    return {"f1": f1, "acc": acc}





