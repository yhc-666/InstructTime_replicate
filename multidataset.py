"""
作用: 定义多领域数据加载类，处理时间序列和文本指令的结合，支持多种时间序列数据类型
输入: 
    - 样本列表(包含文本指令、时间序列数据和标签)
    - tokenizer实例
    - 运行模式(训练/测试)
    - 多模态类型标识
输出: 
    - 格式化的数据批次，包含input_ids, attention_masks和labels
示例用法:
    dataset = MultiDataset(samples, tokenizer, mode="train", multi="ecg", encoder_max_length=256)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn_train)
"""

import torch
from torch.utils.data import Dataset
    
class MultiDataset(Dataset):
    """
    功能: 多领域数据集类，用于创建时间序列和文本指令的混合数据集
    属性:
        - samples: 样本列表，每个样本为(text, time_series, _)的三元组
        - tokenizer: MultiTokenizer实例，用于编码文本和时间序列
        - mode: 运行模式，'train'或'test'
        - max_length: 输入序列的最大长度
        - multi: 时间序列数据类型标识('mix', 'geo', 'sleep', 'esr'等)
        - prefix_tokens: 前缀token列表
    作用: 
        加载不同领域的时间序列数据，与文本指令结合，转换为模型可用的格式
    """

    def __init__(
        self,
        samples,
        tokenizer,
        mode: str,
        multi: str,
        encoder_max_length=256,
        prefix_text="",
    ) -> None:
        """
        功能: 初始化多领域数据集
        输入:
            - samples: 样本列表，每个样本包含(text, time_series, _)
            - tokenizer: MultiTokenizer实例
            - mode: 运行模式，'train'或'test'
            - multi: 多模态类型标识('mix', 'geo', 'sleep'等)
            - encoder_max_length: 编码器的最大序列长度，默认256
            - prefix_text: 输入文本的前缀，默认为空
        """
        assert mode in ["train", "test"]
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = encoder_max_length
        self.multi = multi
        self.prefix_tokens = self.tokenizer.encode(prefix_text) if prefix_text else []
    
    def __len__(self):
        """
        功能: 返回数据集中的样本数量
        返回: 整数，样本数量
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        功能: 获取指定索引的样本并处理为模型输入格式
        输入:
            - idx: 样本索引
        输出:
            - 字典，包含处理后的样本数据:
              训练模式: {'input_ids', 'attn_masks', 'label_ids'}
              测试模式: {'input_ids', 'attn_masks', 'label'}
        处理流程:
            1. 从样本中提取文本和时间序列数据
            2. 从文本中分离指令和标签
            3. 根据模式决定输入文本格式
            4. 使用template函数组合时间序列和文本
            5. 添加注意力掩码和标签ID
            6. 应用padding处理
        """
        text, ecg, _ = self.samples[idx]

        dx_index = text.find("information.\n")
        if dx_index != -1:
            label = text[dx_index + 13:]
            text = text[:dx_index + 13]
        else:
            label = ''
        label_ids = self.tokenizer.encode(label)

        if self.mode == "train":
            text = text + label
        else:
            text = text

        input_ids = self.template(ecg * 2.5, text)
        label_ids = [-100] * (len(input_ids) - len(label_ids)) + label_ids
        
        attn_masks = [1] * len(input_ids)
        input_ids, attn_masks = self.padding(input_ids, attn_masks)
        label_ids, _ = self.padding(label_ids, attn_masks)

        if self.mode == "train":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "label_ids": torch.LongTensor(label_ids), 
            }

        elif self.mode == "test":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "label": label,
            }
        
    def template(self, ecg, text):
        """
        功能: 根据模板组合时间序列和文本，构建模型输入
        输入:
            - ecg: 时间序列数据，不同数据类型有不同的shape
            - text: 文本指令
        输出:
            - input_ids: 整数列表，表示组合后的输入token ID
        处理流程:
            1. 根据数据类型(multi)和时间序列形状确定正确的描述和tokenizer
            2. 编码时间序列数据
            3. 将时间序列token与文本token连接
            4. 截断到最大长度
        """
        input_ids = self.prefix_tokens.copy()
        if self.multi == 'mix':
            if ecg.shape == (5000, 12):
                bet_ids = self.tokenizer.encode('Electrocardiogram signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=0)
            elif ecg.shape == (3000, 2): 
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=1)
            elif ecg.shape == (5120, 1):
                bet_ids = self.tokenizer.encode('Industrial equipment signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=2)
            elif ecg.shape == (128, 9):
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=3)
            elif ecg.shape == (4000, 1):
                bet_ids = self.tokenizer.encode('Whale sound signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=4)
            elif ecg.shape == (93, 13):
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=5)
        else:
            if self.multi == 'geo':
                bet_ids = self.tokenizer.encode('Electrocardiogram signals: <BET>')
            elif self.multi == 'sleep':
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
            elif self.multi == 'esr':
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
            else:
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
            ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0))
        text_ids = self.tokenizer.encode('<EET> \n' + text)

        ecg_ids = ecg_ids.tolist()
        ecg_ids = ecg_ids[0]

        input_ids.extend(bet_ids + ecg_ids + text_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[0 : self.max_length]
        
        return input_ids

    def padding(self, input_ids: list, attn_masks: list):
        """
        功能: 对输入序列应用填充(padding)处理
        输入:
            - input_ids: 整数列表，输入token ID
            - attn_masks: 整数列表，注意力掩码
        输出:
            - input_ids: 填充后的输入token ID列表，长度为self.max_length
            - attn_masks: 填充后的注意力掩码列表，长度为self.max_length
        处理流程:
            - 训练模式：在右侧填充(便于自回归学习)
            - 测试模式：在左侧填充(便于生成任务)
        """
        assert len(input_ids) <= self.max_length

        if self.mode == "train":
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            )
            attn_masks = attn_masks + [0] * (self.max_length - len(attn_masks))
        elif self.mode == "dev" or self.mode == "test":
            input_ids = [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            ) + input_ids
            attn_masks = [0] * (self.max_length - len(attn_masks)) + attn_masks
        return input_ids, attn_masks