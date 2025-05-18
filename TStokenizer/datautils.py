import os
import pickle
import numpy as np

def save_datasets(data, path, file_name):
    """
    功能: 保存数据集为numpy格式
    输入:
        - data: 要保存的数据集数组
        - path: 保存目录路径
        - file_name: 保存的文件名
    """
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, file_name), data)

def load_datasets(path, file_name):
    """
    功能: 加载numpy格式的数据集
    输入:
        - path: 数据集所在目录路径
        - file_name: 数据集文件名
    返回值:
        - 加载的numpy数组数据集
    """
    return np.load(os.path.join(path, file_name))

def load_all_data(Path, use_saved_datasets=True):
    """
    功能: 加载训练和测试数据集
    输入:
        - Path: 数据集目录路径
        - use_saved_datasets: 是否使用预先保存的numpy格式数据集
    返回值:
        - tokenizer_train: 训练数据集
        - tokenizer_test: 测试数据集
    流程:
        1. 尝试直接加载已处理的numpy数据集
        2. 如果不存在, 从原始pickle文件加载并处理数据
        3. 保存处理后的数据为numpy格式以加速后续加载
    """
    if use_saved_datasets:
        try:
            # 尝试加载预先保存的numpy格式数据集
            tokenizer_train = load_datasets(Path, 'train.npy')
            tokenizer_test = load_datasets(Path, 'test.npy')
            print(len(tokenizer_train), len(tokenizer_test))

            return tokenizer_train, tokenizer_test
        except IOError:
            print("Saved datasets not found. Processing raw data.")

    # 获取原始数据文件路径
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
	
    samples_train, tokenizer_train = [], []
    samples_test, tokenizer_test = [], []
    
    # 从pickle文件加载原始数据
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    # 处理训练数据
    for sample in samples_train:
        _, ecg, _ = sample  # 提取时间序列数据部分
        tokenizer_train.append(ecg)
    
    # 处理测试数据
    for sample in samples_test:
        _, ecg, _ = sample  # 提取时间序列数据部分
        tokenizer_test.append(ecg)

    # 转换为numpy数组，使用float32类型节省内存
    tokenizer_train = np.array(tokenizer_train, dtype=np.float32)
    tokenizer_test = np.array(tokenizer_test, dtype=np.float32)

    # 保存处理后的数据集以加速后续加载
    save_datasets(tokenizer_train, Path, 'train.npy')
    save_datasets(tokenizer_test, Path, 'test.npy')

    return tokenizer_train, tokenizer_test

if __name__ == "__main__":
    load_all_data('./esr_data')
