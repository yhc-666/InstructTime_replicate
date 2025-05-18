"""
作用: 定义VQVAE核心模型结构
输入: 
    - 时间序列数据(各种领域)
    - 模型参数(隐藏层大小、块数、波长等)
输出: 
    - 编码后的离散token序列
    - 重构信号和相关损失
示例用法:
    # 创建模型
    model = TStokenizer(
        data_shape=(5000, 12),  # 时间序列形状
        hidden_dim=64,         # 隐藏层维度
        n_embed=1024,          # 嵌入向量数量
        block_num=4,           # TCN块数量
        wave_length=32,        # 波长参数
    )
    
    # 前向传播
    reconstructed, diff, token_ids = model(time_series_data)
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_

"""
TCN based on RecBole's implementation
################################################

Reference code:
    - https://github.com/fajieyuan/nextitnet
    - https://github.com/initlisk/nextitnet_pytorch

"""

class TCN(nn.Module):
    """
    功能: 时间卷积网络，用于序列建模
    输入: 
        - x: 时间序列数据，形状为 [batch_size, seq_len, hidden_dim]
    输出:
        - 处理后的特征，形状为 [batch_size, seq_len, hidden_dim]
    参数:
        - d_model: 隐藏层维度
        - block_num: 残差块数量
        - dilations: 扩张因子列表
        - data_shape: 输入数据形状 (seq_len, feat_dim)
    """
    def __init__(self, args=None, **kwargs):
        super(TCN, self).__init__()

        # load parameters info
        if args is not None:
            d_model = args.d_model
            self.embedding_size = args.d_model
            self.residual_channels = args.d_model
            self.block_num = args.block_num
            self.dilations = args.dilations * self.block_num
            self.kernel_size = args.kernel_size
            self.enabel_res_parameter = args.enable_res_parameter
            self.dropout = args.dropout
            self.device = args.device
            self.data_shape = args.data_shape
        else:
            d_model = kwargs['d_model']
            self.embedding_size = kwargs['d_model']
            self.residual_channels = kwargs['d_model']
            self.block_num = kwargs['block_num']
            self.dilations = kwargs['dilations'] * self.block_num
            self.data_shape = kwargs['data_shape']
            self.kernel_size = 3
            self.enabel_res_parameter = 1
            self.dropout = 0.1

        self.max_len = self.data_shape[0]
        print(self.max_len)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation,
                enable_res_parameter=self.enabel_res_parameter, dropout=self.dropout
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        # self.output = nn.Linear(self.residual_channels, self.num_class)
        self.output = nn.Linear(d_model, d_model)
        self.broadcast_head = nn.Linear(d_model, self.data_shape[1])

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        功能: 参数初始化
        输入:
            - module: 网络模块
        """
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, x):
        """
        功能: 前向传播
        输入:
            - x: 输入序列，形状为 [batch_size, seq_len, feat_dim]
        输出:
            - 处理后的特征，形状为 [batch_size, seq_len, hidden_dim]
        """
        # Residual locks
        # x in shape of [(B*T)*L*D]
        dilate_outputs = self.residual_blocks(x)
        x = dilate_outputs
        return self.output(x)


class ResidualBlock_b(nn.Module):
    """
    功能: TCN中的残差块，应用扩张卷积
    输入:
        - x: 特征张量，形状为 [batch_size, seq_len, channel_dim]
    输出:
        - 残差连接后的特征，形状与输入相同
    参数:
        - in_channel: 输入通道数
        - out_channel: 输出通道数
        - kernel_size: 卷积核大小
        - dilation: 扩张因子
        - enable_res_parameter: 是否使用可学习的残差缩放参数
    """
    def __init__(self, in_channel, out_channel, kernel_size=10, dilation=None, enable_res_parameter=False, dropout=0):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

        self.enable = enable_res_parameter
        self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        """
        功能: 残差块的前向传播
        输入:
            - x: 输入特征，形状为 [batch_size, seq_len, embed_size]
        输出:
            - 残差连接后的特征，形状与输入相同
        """
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.dropout1(self.conv1(x_pad).squeeze(2).permute(0, 2, 1))
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.dropout2(self.conv2(out_pad).squeeze(2).permute(0, 2, 1))
        out2 = F.relu(self.ln2(out2))

        if self.enable:
            x = self.a * out2 + x
        else:
            x = out2 + x

        return x
        # return self.skipconnect(x, self.ffn)

    def conv_pad(self, x, dilation):
        """ 
        功能: 对输入进行填充，应用于扩张卷积前
        输入:
            - x: 输入特征，形状为 [batch_size, seq_len, embed_size]
            - dilation: 扩张因子
        输出:
            - 填充后的特征，形状为 [batch_size, embed_size, 1, padded_seq_len]
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad
    

class Quantize(nn.Module):
    """
    向量量化(Vector Quantization, VQ)层
    ===============================================================

    目标
    ----
    • 将连续隐藏向量离散化为 codebook 中最接近的 prototype,
      把 “近似最近邻查表” 嵌入到计算图里。
    • 同时在训练中使用 EMA(Exponential Moving Average) 更新 codebook,
      相当于在线 K-Means。

    输入 / 输出
    -----------
    input : FloatTensor[*, dim]
        任意批量/时序形状，最后一维 = `dim`.
        例如 (B, T, D) ➔ reshape → (B·T, D).

    返回
    ----
    quantize  : FloatTensor[*, dim]
        离散后再用 STE 反投影的向量, shape 与 input 完全一致。
    diff      : torch.scalar
        VQ 额外损失 (codebook MSE + β·commitment MSE)，需加到总 loss。
    embed_ind : LongTensor[*]
        离散索引（每个 token 对应的码字 id) ,shape = input.shape[:-1].

    参数
    ----
    dim      : codeword 向量维度 (= 隐藏维度)
    n_embed  : codebook 大小 (= 码字数)
    decay    : EMA 衰减率 (典型 0.99)
    eps      : 数值稳定项，防零除
    beta     : commitment 损失权重 (典型 0.25)
    """

    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, beta=0.25):
        super().__init__()

        # --------------------------- 常量保存 ----------------------------
        self.dim     = dim          # codeword 维度
        self.n_embed = n_embed      # 码字个数
        self.decay   = decay        # EMA 系数
        self.eps     = eps
        self.beta    = beta

        # --------------------------- 参数初始化 --------------------------
        # codebook: [dim, n_embed]  (列向量 = 单个 codeword)
        embed = torch.randn(dim, n_embed)
        torch.nn.init.kaiming_uniform_(embed)

        # register_buffer => 与模型同设备保存，但不参与梯度
        self.register_buffer("embed",        embed)            # 当前 codebook
        self.register_buffer("cluster_size", torch.zeros(n_embed))  # 记录各码字被选次数 (EMA)
        self.register_buffer("embed_avg",    embed.clone())    # 记录各码字累加和 (EMA)

    # -------------------------------------------------------------------
    # forward: 离散化 + (训练时) EMA 更新
    # -------------------------------------------------------------------
    def forward(self, input):
        """
        input  : [..., dim]
        返回   : quantize[... ,dim], diff, embed_ind[...]
        """

        # --------1. 计算最近邻 codeword---------------------------------
        # 把 * 维全部展平到 batch_dim，方便矩阵运算
        # flatten: [N, dim]  (N = ∏ 输入除 dim 外的元素)
        flatten = input.reshape(-1, self.dim)

        # pairwise L2 距离:
        # ||x||^2 - 2 x·e + ||e||^2
        #   flatten.pow(2).sum(1, keepdim=True)  -> [N,1]
        #   -2 * flatten @ embed                -> [N, n_embed]
        #   embed.pow(2).sum(0, keepdim=True)   -> [1, n_embed]
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed          # (N, n_embed)
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # 取距离最大的 (负号后最大 = 最小距离)
        # embed_ind: [N]
        _, embed_ind = (-dist).max(1)

        # one-hot: [N, n_embed];  dtype 同 flatten (float16/32)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)

        # 把 [N] 索引还原到 input 的 batch / time 形状
        embed_ind = embed_ind.view(*input.shape[:-1])          # [...]

        # 量化结果 (查表) : [..., dim]
        quantize = self.embed_code(embed_ind)

        # ----------------2. 训练模式下用 EMA 更新 codebook --------------
        if self.training:
            # Σ_onehot_k   → 统计每个 codeword 被选次数
            embed_onehot_sum = embed_onehot.sum(0)                  # [n_embed]
            # Σ_x(k)       → 每个 codeword 聚类质心的累加向量
            embed_sum = flatten.transpose(0, 1) @ embed_onehot     # [dim, n_embed]

            # EMA 更新
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )

            # 对 cluster_size 做 Laplace 平滑，再归一化
            n = self.cluster_size.sum()                                       # 总样本数估计
            cluster_size = (
                (self.cluster_size + self.eps) /
                (n + self.n_embed * self.eps) * n
            )                                                                 # [n_embed]

            # new_embed = Σ_x(k) / N_k
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)     # [dim, n_embed]
            self.embed.data.copy_(embed_normalized)

        # ----------------3. 计算 VQ 损失 (MSE) --------------------------
        # codebook loss:   不传梯度到 encoder，只调 codebook
        codebook_loss = (quantize.detach() - input).pow(2).mean()
        # commitment loss: 不传梯度到 codebook，只调 encoder
        commit_loss   = (quantize - input.detach()).pow(2).mean()

        diff = codebook_loss + self.beta * commit_loss      # scalar

        # ----------------4. Straight-Through Estimator -----------------
        # 反向时把 (quantize - input) 当作常量，不影响梯度，
        # 使 encoder 得到 commitment loss 的梯度。
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    # helper: 根据索引取 codeword
    def embed_code(self, embed_id):
        """
        embed_id: LongTensor[...]   (取值范围 0~n_embed-1)
        返回    : FloatTensor[..., dim]
        """
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class Encoder(nn.Module):
    """
    功能: 编码器模块，将输入时间序列编码为隐藏表示
    输入:
        - input: 时间序列数据，形状为 [batch_size, seq_len, feat_num]
    输出:
        - 编码后的特征，形状为 [batch_size, seq_len, hidden_dim]
    参数:
        - feat_num: 输入特征维度
        - hidden_dim: 隐藏层维度
        - block_num: TCN块数量
        - data_shape: 输入数据形状
        - dilations: 扩张因子列表
    """
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 4]):
        super().__init__()
        self.input_projection = nn.Linear(feat_num, hidden_dim)
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,
                          dilations=dilations)

    def forward(self, input):
        """
        功能: 编码器前向传播
        输入:
            - input: 时间序列数据，形状为 [batch_size, seq_len, feat_num]
        输出:
            - 编码后的特征，形状为 [batch_size, seq_len, hidden_dim]
        """
        return self.blocks(self.input_projection(input))


class Decoder(nn.Module):
    """
    功能: 解码器模块，将隐藏表示重构为原始时间序列
    输入:
        - input: 量化后的特征，形状为 [batch_size, seq_len, hidden_dim]
    输出:
        - 重构的时间序列，形状为 [batch_size, seq_len, feat_num]
    参数:
        - feat_num: 输出特征维度
        - hidden_dim: 隐藏层维度
        - block_num: TCN块数量
        - data_shape: 输出数据形状
        - dilations: 扩张因子列表
    """
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 4]):
        super().__init__()
        self.output_projection = nn.Linear(hidden_dim, feat_num)
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,
                          dilations=dilations)

    def forward(self, input):
        """
        功能: 解码器前向传播
        输入:
            - input: 量化后的特征，形状为 [batch_size, seq_len, hidden_dim]
        输出:
            - 重构的时间序列，形状为 [batch_size, seq_len, feat_num]
        """
        return self.output_projection(self.blocks(input))


class TStokenizer(nn.Module):
    """
    功能: 时间序列标记化模型, 也就是拼装好的VQVAE模型
    输入:
        - input: 时间序列数据，形状为 [batch_size, seq_len, feat_dim]
    输出:
        - dec: 重构的时间序列，形状与输入相同
        - diff: 量化损失, 只包含codebook-loss + β·commitment-loss, 不包含重构损失
        - id: 离散token索引, 形状为 [batch_size, seq_len//wave_length]
    参数:
        - data_shape: 输入数据形状 (seq_len, feat_dim)
        - hidden_dim: 隐藏层维度
        - n_embed: codebook大小
        - block_num: TCN块数量
        - wave_length: 沿时间轴的patchsize
    """
    def __init__(
            self,
            data_shape=(5000, 12),
            hidden_dim=64,
            n_embed=1024,
            block_num=4,
            wave_length=32,
    ):
        super().__init__()

        self.enc = Encoder(data_shape[1], hidden_dim, block_num, data_shape)
        self.wave_patch = (wave_length, hidden_dim)
        self.quantize_input = nn.Conv2d(1, hidden_dim, kernel_size=self.wave_patch, stride=self.wave_patch)
        self.quantize = Quantize(hidden_dim, n_embed)
        self.quantize_output = nn.Conv1d(int(data_shape[0] / wave_length), data_shape[0], kernel_size=1)
        self.dec = Decoder(data_shape[1], hidden_dim, block_num, data_shape)
        self.n_embed = n_embed
        self.hidden_dim = hidden_dim

    def get_name(self):
        """
        功能: 获取模型名称
        输出:
            - 模型名称字符串 'vqvae'
        """
        return 'vqvae'

    def forward(self, input):
        """
        功能: VQVAE模型前向传播
        输入:
            - input: 时间序列数据，形状为 [batch_size, seq_len, feat_dim]
        输出:
            - dec: 重构的时间序列，形状与输入相同
            - diff: 量化损失, 只包含codebook-loss + β·commitment-loss, 不包含重构损失
            - id: 离散token索引, 形状为 [batch_size, seq_len//wave_length]
        流程:
            1. 编码器将输入编码为隐藏表示
            2. 对隐藏表示进行分段和压缩
            3. 量化模块将连续特征转换为离散码本索引
            4. 解码器重建原始序列
        """
        enc = self.enc(input) # (B,   L,   D)  
        # patching: 通过2D conv 进行, 把 时间维 和 隐藏通道维 当作一张 2-D “图片” 的高×宽来切 patch
        enc = enc.unsqueeze(1) # (B, 1, L,   D)  ← N,C,H,W  视角
        quant = self.quantize_input(enc)  # (B, D, L/32, 1) ← 每个 filter = 一个 32×64 patch
        quant = quant.squeeze(-1) # (B, D, L/32)
        quant = quant.transpose(1, 2) # (B, L/32,   D)  ← token 序列 
        quant, diff, id = self.quantize(quant) 
        quant = self.quantize_output(quant) 
        dec = self.dec(quant) 

        return dec, diff, id
    
    def get_embedding(self, id):
        """
        功能: 根据token索引获取对应的特征向量
        输入:
            - id: token索引, 形状为 [batch_size, seq_len//wave_length]
        输出:
            - 对应的特征向量，形状为 [batch_size, seq_len//wave_length, hidden_dim]
        """
        return self.quantize.embed_code(id)
    
    def decode_ids(self, id):
        """
        功能: 直接从token索引解码重构时间序列
        输入:
            - id: token索引, 形状为 [batch_size, seq_len//wave_length]
        输出:
            - dec: 重构的时间序列，形状为 [batch_size, seq_len, feat_dim]
        """
        quant = self.get_embedding(id)
        quant = self.quantize_output(quant)  # [batch_size, seq_len//wave_length]→[batch_size, seq_len, hidden_dim]
        dec = self.dec(quant)

        return dec

if __name__ == '__main__':
    model = TStokenizer()
    a = torch.randn(2, 5000, 12)
    tmp = model(a)
    print(1)
