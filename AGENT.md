## Your Task

这是论文instrctime的代码repo。

现在我想要在这个repo上跑通我自己的ihm+phe benchmark。

我的想法是第一步先**把 IHM + PHE 数据集中的时间序列部分全部一起喂 VQ-Encoder，训练一个**“通用 ICU codebook”。

超参数设置：

| 设置       | 建议值    | 说明                                        |
| ---------- | --------- | ------------------------------------------- |
| patch_len  | 4 h       | 48 h→12 patch，24 h→6 patch，不会拉长 token |
| codebook K | 256       | ICU 指标只有 17 维，256 足够表达            |
| epoch × lr | 30 × 1e-3 | 仅 34→64 MLP + TCN，显存≈2 GB；< 30 min     |

**产物**：`vq.ckpt` + Python callable `vq.encode(ts)`，把 (T,34) → token_seq。



请协助我对应的修改相关文件。



## 我自己的数据集格式、

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