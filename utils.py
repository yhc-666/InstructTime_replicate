"""
作用: 提供InstructTime项目通用工具函数，包括模型加载、文本处理和辅助函数
输入: 
    - 各种需要处理的数据和文件路径
    - 模型参数和配置
输出: 
    - 加载好的模型组件
    - 处理后的文本和数据
示例用法:
    # 加载时间序列tokenizer
    tokenizer = load_TStokenizer(dir_path="./model_path", data_shape=(5000, 12), device="cuda:0")
    
    # 从文本中提取信息
    diagnosis, stage, har, fd, rwc = extract_all_information(text)
"""

import os
import torch
import json
import random
from TStokenizer.model import TStokenizer

def get_fixed_order_choice(labels):
    """
    功能: 随机打乱标签列表并返回
    输入:
        - labels: 标签列表，例如可能的疾病诊断或活动类别
    输出:
        - 打乱顺序后的标签列表
    用途:
        用于生成选择题选项或增加数据多样性
    """
    shuffled_labels = labels[:]
    shuffled_labels = list(shuffled_labels)
    random.shuffle(shuffled_labels) 
    return shuffled_labels

def extract_all_information(text):
    """
    功能: 从文本中提取所有类型的关键信息
    输入:
        - text: 文本字符串，可能包含各种领域的预测结果
    输出:
        - diagnosis: ECG诊断结果字符串
        - stage: EEG阶段分类结果字符串
        - har: 人体活动识别结果字符串
        - dev: 设备故障分类结果字符串
        - whale: 鲸鱼声音分类结果字符串
    处理流程:
        根据文本中的关键词标识，提取相应类型的信息
    """
    diagnosis = stage = har = dev = whale = ""
    if "include(s)" in text:
        diagnosis = extract_from_text(text, "include(s) ")
    elif "pattern is" in text:
        stage = extract_from_text(text, "pattern is ")
    elif "engaged in" in text:
        har = extract_from_text(text, "engaged in ")
    elif "conditions:" in text:
        dev = extract_from_text(text, "conditions: ")
    elif "originates from" in text:
        whale = extract_from_text(text, "originates from ")
    return diagnosis, stage, har, dev, whale

def extract_from_text(text, keyword):
    """
    功能: 从文本中提取关键词后面的内容
    输入:
        - text: 文本字符串
        - keyword: 关键词字符串，用于定位提取位置
    输出:
        - 关键词后面的文本内容，如果找不到关键词则返回空字符串
    """
    index = text.find(keyword)
    if index != -1:
        return text[index + len(keyword):] 
    return ""

def load_params_from_json(json_file_path):
    """
    功能: 从JSON文件加载模型参数
    输入:
        - json_file_path: JSON文件路径
    输出:
        - 包含参数的字典，键为参数名，值为参数值
    """
    with open(json_file_path, 'r') as file:
        params = json.load(file)
    return params

def load_TStokenizer(dir_path, data_shape, device):
    """
    功能: 加载时间序列tokenizer模型
    输入:
        - dir_path: 模型目录路径, 包含args.json和model.pkl文件
        - data_shape: 时间序列数据形状，如(5000, 12)表示ECG数据
        - device: 设备标识，如'cuda:0'或'cpu'
    输出:
        - 加载好的TStokenizer模型实例, 已设置为评估模式
    处理流程:
        1. 加载模型参数
        2. 初始化TStokenizer模型
        3. 加载预训练权重
        4. 将模型设置为评估模式并移到指定设备
    """
    json_params_path = os.path.join(dir_path, "args.json")
    model_path = os.path.join(dir_path, "model.pkl")

    params = load_params_from_json(json_params_path)

    vqvae_model = TStokenizer(data_shape=data_shape, hidden_dim=params['hidden_dim'], n_embed=params['n_embed'], wave_length=params['wave_length'])
    vqvae_model.load_state_dict(torch.load(model_path, map_location=device))
    vqvae_model.eval()
    vqvae_model = vqvae_model.to(device)  # 确保模型移动到指定设备
    
    return vqvae_model
