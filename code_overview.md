# InstructTime代码库概览

## 架构速览

```
InstructTime/
├── .git/                          # Git版本控制目录
├── TStokenizer/                   # 时间序列向量化模块
│   ├── args.py                    # 命令行参数定义
│   ├── dataset.py                 # 数据集类定义与加载
│   ├── datautils.py               # 数据处理工具函数
│   ├── loss.py                    # 损失函数实现
│   ├── main.py                    # TStokenizer训练入口
│   ├── main_eval.py               # TStokenizer评估脚本
│   ├── model.py                   # VQVAE模型和TCN实现
│   └── process.py                 # 训练和评估流程
├── args.py                        # 主模型参数配置
├── challeng_score.py              # 挑战评分计算
├── metrics.py                     # 评估指标计算
├── multimodel.py                  # InstructTime核心模型实现
├── multidataset.py                # 跨域数据集加载器
├── preprocess.py                  # 数据预处理与增强
├── requirements.txt               # 依赖包列表
├── run_truth_loss.py              # 模型训练与推理入口
└── utils.py                       # 通用工具函数
```

## 关键组件说明

1. **TStokenizer**: 时间序列编码器，将各种时间序列数据转化为离散的tokens
   - 基于VQVAE (Vector Quantized Variational Autoencoder)设计
   - 使用TCN (Temporal Convolutional Networks)进行特征提取
   - 不同领域数据有专门的tokenizer实例

2. **InstructTime模型**: 多模态模型，结合时间序列token和文本指令
   - 基于GPT2架构，扩展为处理时间序列token
   - 支持多种时间序列数据类型(ECG、EEG、HAR等)
   - 实现了Universal版本(通用领域)和Adapt版本(领域微调)

3. **数据处理流程**:
   - 原始时间序列数据预处理
   - VQVAE编码为离散tokens
   - 与文本指令组合训练语言模型
   - 支持多领域任务

4. **训练与评估**:
   - 两阶段训练：先训练TStokenizer，再训练InstructTime
   - 使用特定领域的评估指标
   - 支持多领域联合训练和单域微调 



## VQVAE

```
 ┌───────────────┐
 │ Raw time-series│  (B, 5000, 12)
 └──────┬────────┘
        │Linear 12→64
        ▼
 ┌───────────────┐
 │ Temporal CNN  │  (B, 5000, 64)
 │  (dilated TCN)│
 └──────┬────────┘
        │unsqueeze & patch-conv2d
        ▼
 ┌───────────────┐
 │ token tensor  │  (B, 156, 64)   ← ⌊5000/32⌋ = 156
 └──────┬────────┘
        │Vector-Quantization
        ▼
 ┌───────────────┐
 │ quantized rep │  (B, 156, 64)
 │  + code indices───►(B,156)
 └──────┬────────┘
        │Conv1d up-sample
        ▼
 ┌───────────────┐
 │ hidden series │  (B, 5000, 64)
 └──────┬────────┘
        │Decoder (TCN)
        ▼
 ┌───────────────┐
 │ Reconstructed │  (B, 5000, 12)
 └───────────────┘

```





