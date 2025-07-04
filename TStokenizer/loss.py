import torch.nn as nn

"""
重构 + latent 损失
"""

class MSE:
    """
    功能: 计算VQVAE模型的损失，包括重构损失和latent损失
    该类组合了重构损失(MSE)和codebook承诺损失(commitment loss)
    """
    def __init__(self, model, latent_loss_weight=0.35):
        """
        功能: 初始化MSE损失计算器
        输入:
            - model: VQVAE模型实例
            - latent_loss_weight: latent损失权重，用于平衡重构损失和latent损失
        """
        self.model = model
        self.latent_loss_weight = latent_loss_weight
        self.mse = nn.MSELoss()

    def compute(self, batch, mask=None):
        """
        功能: 计算模型的总损失
        输入:
            - batch: 输入数据批次，形状为 [batch_size, seq_len, feat_dim]
        输出:
            - loss: 总损失值，包括重构损失和加权latent损失
        计算流程:
            1. 模型前向传播得到重构输出、latent损失和token索引
            2. 计算重构损失(原始输入与重构输出的MSE)
            3. 计算latent损失均值
            4. 组合重构损失和加权latent损失
        """
        seqs = batch
        # 前向传播，得到重构结果、latent损失和token索引
        out, latent_loss, _ = self.model(seqs, mask=mask)

        if mask is not None:
            recon_loss = ((out - seqs) ** 2 * mask).sum() / mask.sum()
        else:
            recon_loss = self.mse(out, seqs)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss
        return loss
