<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Replication of InstructTime: Advancing Time Series Classification with Multimodal Language Modeling (WSDM2025) </b></h2>
</div>

## Project Overview
This is the github repo for the paper InstructTime. I am adapting it to MIMIC3 48h phenotype classification and 24h ihm prediction benchmark.

## 数据集格式

数据集位于 `/Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/output_mimic3`

首先我在预处理后得到ihm/pheno 分别的下游prediction任务数据集，每个数据集被切分为train/val/test，以pkl格式存储于`PATH/ihm、PATH/pheno`文件夹中

1. 48h ihm prediction：对应`train_p2x_data.pkl`, `val_p2x_data.pkl`, `test_p2x_data.pkl`,一共14066条

| 字段名        | 数据类型            | Shape / 长度示例                  | 描述                                                         | 含义说明                                  |
| ------------- | ------------------- | --------------------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| `reg_ts`      | `numpy.ndarray`     | `(48, 34)`                        | 规则化后的多变量时间序列（每小时一次，共 48 小时）。前 17 列为数值特征（生命体征、检验指标等），后 17 列为对应的特征缺失掩码 | 结构化、均匀采样处理后的时序输入          |
| `irg_ts`      | `numpy.ndarray`     | `(len, 17)`, len的大小不固定      | 原始不规则采样的多变量时间序列，包含缺失                     | 真实测量值的时间序列输入(未经48h对齐处理) |
| `irg_ts_mask` | `numpy.ndarray`     | `(len, 17)`, len的大小不固定      | 同上                                                         | 缺失标记（1 = 存在，0 = 缺失）            |
| `ts_tt`       | `list[float]`       | `length = len`                    | 与 `irg_ts` 行对应的时间戳（单位：小时）                     | 每条时序测量的发生时间                    |
| `text_data`   | `list[str]`         | 长度不固定                        | 临床自由文本（护理/病程记录等）                              | 非结构化文本输入                          |
| `text_time`   | `list[float]`       | `length = len(text_data)`         | 文本对应的时间戳（小时）                                     | 文本事件发生时间                          |
| `label`       | `list[int]` / `int` | IHM: `()`                         | 预测标签：院内死亡（二分类 0/1）                             | 监督信号                                  |
| `name`        | `str`               | 例: 10163_episode1_timeseries.csv | 样本文件名或唯一标识                                         | 便于追溯与调试                            |

2. 24h phenotype classification, 对应`train_p2x_data.pkl`, `val_p2x_data.pkl`, `test_p2x_data.pkl`,一共23163条

| 字段名        | 数据类型            | Shape / 长度示例               | 描述                                                         | 含义说明                       |
| ------------- | ------------------- | ------------------------------ | ------------------------------------------------------------ | ------------------------------ |
| `reg_ts`      | `numpy.ndarray`     | `(24, 34)`                     | 规则化后的多变量时间序列（每小时一次，共 24 小时）。前 17 列为数值特征（生命体征、检验指标等），后 17 列为对应的特征缺失掩码 | 结构化、均匀采样后的时序输入   |
| `irg_ts`      | `numpy.ndarray`     | ``(len, 17)`, len的大小不固定` | 原始不规则采样的多变量时间序列，包含缺失                     | 真实测量值的时间序列输入       |
| `irg_ts_mask` | `numpy.ndarray`     | ``(len, 17)`, len的大小不固定` | 同上                                                         | 缺失标记（1 = 存在，0 = 缺失） |
| `ts_tt`       | `list[float]`       | `length = len`                 | 与 `irg_ts` 行对应的时间戳（单位：小时）                     | 每条时序测量的发生时间         |
| `text_data`   | `list[str]`         | 长度不固定                     | 临床自由文本（护理/病程记录等）                              | 非结构化文本输入               |
| `text_time`   | `list[float]`       | ``length = len(text_data)``    | 文本对应的时间戳（小时）                                     | 文本事件发生时间               |
| `label`       | `list[int]` / `int` | PHE: `(25,)`                   | 预测标签：25 类表型多标签（0/1 × 25）                        | 监督信号                       |
| `name`        | `str`               | —                              | 样本文件名或唯一标识                                         | 便于追溯与调试                 |

## Installation dependencies

```bash
pip install -r requirements.txt
```

## Train ICU VQVAE TStokenizer
训练一个VQVAE时间序列tokenizer训练流程将时间序列数据转换为离散token，为后续的LLM训练做准备。

关键步骤：
1. 加载并预处理时间序列数据
2. 初始化VQVAE模型
3. 训练模型并压缩时间序列为离散token
4. 保存训练好的模型

```bash
python train_icu_codebook.py --data_root PATH_TO_DATA --device cuda:0 \
    --save_path ./vq.ckpt --epochs 30 --lr 1e-3
```

## Tokenize original TS
使用训练好的VQVAE TS tokenizer提前将原始数据集中的TS tokenize为token ids, 并保存为pkl文件。
加速后续训练，避免每次重新tokenize。

```bash
python preprocess_tokenize.py \
   --data_root   PATH_TO_DATA \
   --tokenizer   PATH_TO_vqvae_weight \
   --device      cpu \
```

## LLM自回归训练

LLM自回归训练是在预训练的GPT2模型基础上，结合时间序列tokenizer的能力，使模型能够同时理解文本和时间序列数据。

### 关键步骤

1. **准备数据**
   - 加载预处理后的文本和时间序列数据
   - 使用`MultiDataset`类处理混合数据
   - 创建训练和测试数据加载器

2. **初始化模型**
   - 加载预训练的GPT2模型
   - 使用`InstructTime`类集成GPT2和时间序列tokenizer
   - 扩展词表以支持时间序列token

3. **训练模型**
   - 定义优化器、调度器和损失函数
   - 使用`train_model`函数进行自回归训练
   - 训练过程中监控损失函数变化

### 涉及的主要文件和函数

- **train_ar.py**：自回归预训练脚本
  - `initialize_model`：初始化 InstructTime 模型
  - `train_model`：训练循环
  - `validate_ar`：验证评估函数

- **multimodel.py**：模型定义
  - `InstructTime`：主模型类，继承自GPT2
  - `MultiTokenizer`：处理文本和时间序列token的tokenizer类

- **multidataset.py**：数据处理
  - `MultiDataset`：处理混合数据的数据集类
  - `template`：构建输入模板的函数

- **utils.py**：工具函数
  - `load_TStokenizer`：加载时间序列tokenizer
  - `extract_all_information`：从文本中提取信息

### 执行命令示例

```bash
# 训练混合(MIX)数据集的自回归模型
python train_ar.py \
  --device "cuda:0" \
  --dataset "mix" \
  --batch_size 16 \
  --lr 1e-5 \
  --epochs 15

# 仅在 IHM 数据上训练自回归模型
python train_ar.py \
  --device "cuda:0" \
  --dataset "ihm" \
  --batch_size 16 \
  --lr 1e-5 \
  --epochs 10
```

## 下游SFT

下游指令微调在自回归预训练后的模型基础上，使用 `train_sft.py` 脚本针对特定任务进行微调，复用了相同的数据处理与模型模块。

### 关键步骤

1. **准备指令数据**
   - 构建指令-响应对数据集
   - 设计针对时间序列分析任务的指令模板
   - 创建指令微调的数据加载器

2. **加载预训练模型**
   - 加载自回归训练后的模型权重
   - 配置模型以适应指令微调任务

3. **微调模型**
   - 使用较小的学习率进行微调
   - 应用梯度累积和混合精度训练技术
   - 在每个epoch后评估模型性能

4. **模型评估和保存**
   - 使用特定领域的评估指标评估模型
   - 保存性能最佳的模型版本
   - 进行模型推理测试

### 执行命令示例

```bash
# 在 MIX 数据集上进行指令微调
python train_sft.py \
  --device "cuda:0" \
  --dataset "mix" \
  --init_model_path "./gptmodel/no_frozen/run_0/best_model" \
  --batch_size 8 \
  --lr 5e-6 \
  --epochs 5

# 在 IHM 数据集上进行指令微调
python train_sft.py \
  --device "cuda:0" \
  --dataset "ihm" \
  --init_model_path "./pretrained_ar_model" \
  --batch_size 8 \
  --lr 5e-6 \
  --epochs 5
```


## Prompt exaples
**IHM example**

```

Given the 48-hour ICU multivariate vital-sign sequence and doctors' notes below, predict whether the patient will die in hospital.
The possible outcomes are: the patient will survive; the patient will die.

Doctors' notes:
[Note-1] 
…
[Note-K] 

Vital-sign time-series:
<BET> <TS_TOKENS> <EET>.

Answer Output:

The patient will die.
```

**Pheno example**
```
Given the 24-hour ICU multivariate vital-sign sequence and doctors' notes below, select all phenotypes that apply to this ICU stay and briefly state them.
The 25 possible phenotypes include:
Acute and unspecified renal failure,
Acute cerebrovascular disease,
Acute myocardial infarction,
Cardiac dysrhythmias,
Chronic kidney disease,
Chronic obstructive pulmonary disease and bronchiectasis,
Complications of surgical procedures or medical care,
Conduction disorders,
Congestive heart failure; nonhypertensive,
Coronary atherosclerosis and other heart disease,
Diabetes mellitus with complications,
Diabetes mellitus without complication,
Disorders of lipid metabolism,
Essential hypertension,
Fluid and electrolyte disorders,
Gastrointestinal hemorrhage,
Hypertension with complications and secondary hypertension,
Other liver diseases,
Other lower respiratory disease,
Other upper respiratory disease,
Pleurisy; pneumothorax; pulmonary collapse,
Pneumonia (except that caused by tuberculosis or sexually transmitted disease),
Respiratory failure; insufficiency; arrest (adult),
Septicemia (except in labor),
Shock.

Doctors' notes:
[Note-1] 
…
[Note-K] 

Vital-sign time-series:
<BET> <TS_TOKENS> <EET>.

Answer Output:

The patient presents … and … and ... 
```