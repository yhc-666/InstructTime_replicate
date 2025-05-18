import torch
import torch.utils.data as Data
from args import Train_data, Test_data

class Dataset(Data.Dataset):
    """
    功能: 时间序列数据集类，用于训练和测试VQVAE模型
    继承自torch.utils.data.Dataset
    """
    def __init__(self, device, mode, args):
        """
        功能: 初始化数据集
        输入:
            - device: 计算设备(CPU/GPU)
            - mode: 'train'或'test'，指定使用训练集或测试集
            - args: 参数配置对象
        """
        self.args = args
        # 根据mode加载相应的数据集
        if mode == 'train':
            self.ecgs_images = Train_data  # 加载训练数据
        else:
            self.ecgs_images = Test_data   # 加载测试数据
        self.device = device  # 设备(CPU/GPU)
        self.mode = mode      # 数据集模式

    def __len__(self):
        """
        功能: 返回数据集大小(样本数量)
        返回值:
            - 数据集中的样本数量
        """
        return len(self.ecgs_images)

    def __getitem__(self, item):
        """
        功能: 获取指定索引的数据样本
        输入:
            - item: 样本索引
        返回值:
            - 处理后的时间序列数据，已转换为tensor并移动到指定设备
            - 注意数据乘以2.5进行了放大处理
        """
        ecg_img = torch.tensor(self.ecgs_images[item]).to(self.device)
        return ecg_img * 2.5  # 对数据进行放大处理

    def shape(self):
        """
        功能: 获取单个样本的形状
        返回值:
            - 数据样本的形状 (seq_len, feat_dim)
        """
        return self.ecgs_images[0].shape
