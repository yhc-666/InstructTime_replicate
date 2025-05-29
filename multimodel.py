"""
作用: 定义InstructTime主模型架构，结合预训练的GPT2模型和时间序列tokenizer创建多模态模型
输入: 
    - GPT2Config配置
    - 时间序列tokenizer列表
    - 文本embedding大小
输出: 
    - InstructTime模型实例，能够同时处理文本和时间序列token
示例CLI:
    python run_truth_loss.py --device "cuda:0" --dataset "mix" --batch_size 16 --lr 1e-5 --epochs 10
"""

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import AutoTokenizer 

local_model_path = "./gpt2-model"

class MLP(nn.Module):
    """
    功能: 多层感知机网络, 用于时间序列token的特征投影
    属性:
        - linear_layers: 线性层列表，包含多层线性变换
    作用: 
        将时间序列Tokenizer的embedding投影到GPT2模型的embedding空间
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        功能: 初始化MLP网络
        输入:
            - input_dim: 输入特征维度, 与时间序列tokenizer的隐藏层维度一致
            - hidden_dims: 隐藏层维度列表, 如[64, 128, 256, 512]
            - output_dim: 输出特征维度, 与GPT2模型的embedding维度一致
        """
        super(MLP, self).__init__()

        all_dims = [input_dim] + hidden_dims + [output_dim]

        self.linear_layers = nn.ModuleList()
        for i in range(len(all_dims) - 1):
            self.linear_layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))

    def forward(self, x):
        """
        功能: 前向传播，对输入特征进行多层变换
        输入:
            - x: 输入特征张量, shape=[batch_size, seq_len, input_dim]
        输出:
            - x: 输出特征张量, shape=[batch_size, seq_len, output_dim]
        """
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < len(self.linear_layers) - 1:
                x = F.gelu(x)
        return x
    
    
class InstructTime(GPT2LMHeadModel):
    """
    功能: InstructTime主模型, 继承自GPT2LMHeadModel, 增加时间序列处理能力
    属性:
        - ecgTokenizers: 时间序列tokenizer列表
        - embed_layer: 共享的时间序列embedding层
        - projection_layers: 时间序列embedding投影层列表
        - offsets: 各tokenizer的token ID偏移量
    作用: 
        1. 整合GPT2和时间序列tokenizer能力
        2. 处理混合输入(文本token和时间序列token)
        3. 支持指令微调和自回归生成
    """
    def __init__(self, config, ecgTokenizers, text_embedding=50258):
        """
        功能: 初始化InstructTime模型
        输入:
            - config: GPT2Config实例, 模型配置
            - ecgTokenizers: 时间序列tokenizer列表
            - text_embedding: 文本词表大小, 默认为50258
        """
        super().__init__(config)
        self.ecgTokenizers = ecgTokenizers

        # 创建时间序列tokenizer的共享embedding层(不同domain/tokenizer share同一个embedding层)
        """
        embed_vector
        1.定义:
            它是一个临时变量,初始创建为空张量torch.empty(0, self.ecgTokenizers[0].hidden_dim),用于收集所有时间序列tokenizer的嵌入向量。
        2.shape:
            [n_embed_1 + n_embed_2 + ... + n_embed_n, hidden_dim]
            其中：
                -n_embed_i 是第i个tokenizer的码本大小(每个domain对应一个tokenizer)
                -hidden_dim 是VQVAE/codebook向量的隐藏层维度
        3.构建过程:
            从每个时间序列tokenizer中提取quantize.embed(即码本/codebook)
            将其转置后拼接到embed_vector中
            这个向量包含了所有时间序列tokenizer的码字(codeword)
        4.作用:
            它是创建embed_layer的基础数据,包含了所有可能的时间序列token的特征表示。
        
        embed_layer
        1.定义: 
            self.embed_layer = nn.Embedding.from_pretrained(embed_vector),它是从embed_vector预初始化的嵌入层。
        2.功能:
            将时间序列token ID映射为对应的嵌入向量
            提供了一个查找表机制,根据token ID直接获取其对应的特征表示
            整合了所有时间序列tokenizer的码本(codebook)到一个统一的嵌入空间
        3.工作流程:
            在前向传播中,当模型接收到混合输入(包含文本token和时间序列token)时
            对于时间序列token,通过self.embed_layer(tokenizer_ids)查询其特征表示
            然后通过projection_layers将这些特征投影到与GPT2模型相同的嵌入维度空间
        """
        embed_vector = torch.empty(0, self.ecgTokenizers[0].hidden_dim)
        for tokenizer in self.ecgTokenizers:
            tokenizer_embed_vector = copy.deepcopy(tokenizer.quantize.embed).transpose(-1, 0)
            embed_vector = torch.cat([embed_vector, tokenizer_embed_vector], dim=0)
        self.embed_layer = nn.Embedding.from_pretrained(embed_vector)   # TS embedding层(由各domain的codebook拼接而成)
        
        self.text_embedding = text_embedding
        self.embed = config.n_embd
        self.config.pad_token_id = self.config.eos_token_id if self.config.pad_token_id is None else self.config.pad_token_id

        # 为不同的domain/tokenizer分别创建不同的投影层
        self.projection_layers = nn.ModuleList()
        for _ in ecgTokenizers:
            mlp = MLP(self.ecgTokenizers[0].hidden_dim, [64, 128, 256, 512], self.embed)
            mlp.apply(self.init_weights_kaiming)
            self.projection_layers.append(mlp)

        # 为不同的domain/tokenizer分别创建不同的token ID offset
        self.offsets = [self.text_embedding] # initial offset = [50258]
        for tokenizer in self.ecgTokenizers:
            self.offsets.append(self.offsets[-1] + tokenizer.n_embed)

    @staticmethod
    def init_weights_kaiming(m):
        """
        功能: 使用Kaiming初始化方法初始化线性层权重
        输入:
            - m: 模型层
        """
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, *args, **kwargs):
        """
        功能: 模型前向传播，处理混合输入
        输入:
            - input_ids: 输入token ID, shape=[batch_size, seq_len] (其中TS tokens的ID已经包含了offset)
            - attention_mask: 注意力掩码, shape=[batch_size, seq_len]
            - labels: 目标token ID, shape=[batch_size, seq_len]
        输出:
            - outputs: 模型输出, 包含logits和loss
        
        pipeline:
            D=config.n_embd=GPT-2 嵌入维度（默认 768)
            H=ecgTokenizers[0].hidden_dim,VQ-VAE 码本维度
            V=text_embedding+∑n_embed,联合词表大小
                            ┌─────────────────────────────────────────────┐
            input_ids  ───► │ token 归属判别 text_mask / ecg_mask          │
                            └┬────────────────────────────────────────────┘
                            │                                           │
                            │text token                               TS token
                            ▼                                           ▼
                    ┌──────────────────┐                       ┌──────────────────┐
                    │ GPT-2 word embed │                       │ shared TS embed  │
                    │  wte (50258*D)   │                       │  (∑n_embed*H)    │
                    └──────────────────┘                       └──────────────────┘
                            │                                          │
                            │                                          │
                            ▼   zero-out padding                       ▼
                    text_embeddings  ──────────────────►  projection_layers[i]
                    (B*L*D)                               (H → D) MLP
                                                            │
                                                            ▼
                                                    ecg_embeddings_i (B*L*D)
                                                            │
                                            sum over i ─────┘
                            └───────────────► 合并 (B*L*D) ◄──────────────────┘
                                            inputs_embeds
                                                        │
                                                        ▼
                                    GPT-2 Transformer (12 L, 768 D)
                                                        │
                                                        ▼
                                                logits (B*L*V)
        """
        input_ids = kwargs["input_ids"]

        # 区分文本token和时间序列token
        text_mask = torch.lt(input_ids, self.text_embedding)
        ecg_mask = ~text_mask
        
        # 处理文本token
        text_ids = input_ids.clone()
        text_ids[ecg_mask] = self.config.pad_token_id
        
        text_embeddings = self.transformer.wte(text_ids)
        text_embeddings.mul_(text_mask.float().unsqueeze(-1))

        # 处理时间序列token
        masked_ids = input_ids.clone()
        masked_ids[text_mask] = 0
        masked_ids[ecg_mask] -= self.text_embedding

        ecg_embeddings = torch.zeros_like(text_embeddings)
        for i, _ in enumerate(self.ecgTokenizers):
            # forward中输入的input_ids已经包含了offset, 而不是domain codebook内的索引id
            tokenizer_mask = (input_ids >= self.offsets[i]) & (input_ids < self.offsets[i + 1]) 
            tokenizer_ids = input_ids.clone()
            tokenizer_ids[~tokenizer_mask] = 0
            tokenizer_ids[tokenizer_mask] -= self.text_embedding

            tokenizer_embeddings = self.embed_layer(tokenizer_ids)
            tokenizer_embeddings = self.projection_layers[i](tokenizer_embeddings)
            tokenizer_embeddings.mul_(tokenizer_mask.float().unsqueeze(-1))
            ecg_embeddings.add_(tokenizer_embeddings)

        # 合并embedding并传入GPT2模型
        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = ecg_embeddings + text_embeddings 
        
        outputs = super().forward(*args, **kwargs)
        return outputs    

    @classmethod
    def from_pretrained(cls, model_path, ecgTokenizers, text_embedding=50258, device="cpu", **kwargs):
        """
        功能: 从保存的模型路径加载完整的InstructTime模型
        输入:
            - model_path: 保存模型的路径，包含config.json和模型权重文件
            - ecgTokenizers: 时间序列tokenizer列表
            - text_embedding: 文本词表大小
            - device: 设备
        输出:
            - 加载完整权重的InstructTime模型实例
        """
        import os
        import json
        from transformers import GPT2Config
        
        # 加载配置
        config = GPT2Config.from_pretrained(model_path)
        
        # 创建模型实例
        model = cls(config, ecgTokenizers, text_embedding)

        # 创建MultiTokenizer实例以获取正确的词表大小
        multi_tokenizer = MultiTokenizer(ecgTokenizers)

        # ① 由于新增 <BET>, <EET> 两个文本 token，先把 wte (50259→50261)
        #    也就是 nn.Embedding(vocab_txt, hidden) 扩容。
        #    新行随机 ~ N(0, config.initializer_range²)；lm_head 会一起扩且仍与 wte 绑权重。
        model.resize_token_embeddings(len(multi_tokenizer.textTokenizer))

        # ② 保留"扩好后仍 tied 的 lm_head"权重视图，稍后拷贝到自定义更大 lm_head。
        current_output = model.get_output_embeddings()   # shape (50261, hidden)

        # ③ 新建一个更大的的 lm_head：行 = 文本 50261 + ΣK_i(codebooks)，列 = hidden
        new_output = nn.Linear(config.n_embd, multi_tokenizer.vocabSize_all(), bias=False).to(device)

        # ④ 继承文本部分权重；codebook 行保持随机初始化
        new_output.weight.data[: len(multi_tokenizer.textTokenizer)] = current_output.weight.data

        # ⑤ 替换输出投影，解除 wte 与 lm_head 的 weight-tying
        model.set_output_embeddings(new_output)
        #    - 输入端 wte 只覆盖 0-50260 的文本 ID
        #    - 输出端可预测文本 ID + 各 codebook token
        #    - codebook token 的输入嵌入由 TSEmbedding 负责

        model.config.vocab_size = multi_tokenizer.vocabSize_all()
        
        # 尝试加载模型权重，支持多种格式
        model_weights_paths = [
            os.path.join(model_path, "model.safetensors"),
            os.path.join(model_path, "pytorch_model.bin"),
            os.path.join(model_path, "model.bin")
        ]
        
        state_dict = None
        loaded_path = None
        
        for weights_path in model_weights_paths:
            if os.path.exists(weights_path):
                loaded_path = weights_path
                if weights_path.endswith('.safetensors'):
                    try:
                        from safetensors.torch import load_file
                        state_dict = load_file(weights_path, device=device)
                        print(f"Loading model weights from safetensors: {weights_path}")
                    except ImportError:
                        print("Warning: safetensors not installed, trying to load as pytorch format")
                        continue
                else:
                    state_dict = torch.load(weights_path, map_location=device)
                    print(f"Loading model weights from pytorch format: {weights_path}")
                break
        
        if state_dict is None:
            raise FileNotFoundError(f"Model weights file not found in {model_path}")
        
        # 加载权重到模型
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading InstructTime model: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading InstructTime model: {unexpected_keys}")
            
        print(f"Successfully loaded complete InstructTime model from {loaded_path}")
            
        model.to(device)
        return model



class MultiTokenizer:
    """
    功能: 多模态tokenizer封装类, 整合文本tokenizer和多个时间序列tokenizer
    属性:
        - textTokenizer: GPT2Tokenizer实例, 处理文本
        - ecgTokenizers: 时间序列tokenizer列表
        - text_vocab_size: 文本词表大小
        - offsets: 各tokenizer的token ID偏移量列表
    作用: 
        提供统一的编码接口，可以处理文本或时间序列输入
    """
    def __init__(self, ecgTokenizers, base_model: str = "gpt2") -> None:
        """
        功能: 初始化MultiTokenizer
        输入:
            - ecgTokenizers: 时间序列tokenizer列表
            - base_model: 基础模型名称，默认为"gpt2"
        """
        if base_model.lower() == "llama3":
            # self.textTokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            self.textTokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                use_fast=True,              # 强制用 rust backend
                trust_remote_code=False,
            )
        else:   
            # default gpt2
            self.textTokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        new_special_tokens = ["<BET>", "<EET>"]
        self.textTokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
        
        # 设置PAD token为EOS token（GPT2的标准做法）
        self.textTokenizer.pad_token = self.textTokenizer.eos_token
        
        self.text_vocab_size = len(self.textTokenizer)
        self.ecgTokenizers = ecgTokenizers

        self.pad_token_id = self.textTokenizer.pad_token_id  # 50256
        self.eos_token_id = self.textTokenizer.eos_token_id  # 50256

        self.offsets = self._calculate_offsets()

    def _calculate_offsets(self):
        """
        功能: 计算每个tokenizer的token ID偏移量
        返回:
            - offsets: 整数列表, 每个tokenizer的偏移量
        """
        offsets = []
        current_offset = self.text_vocab_size
        for tokenizer in self.ecgTokenizers:
            offsets.append(current_offset)
            current_offset += tokenizer.n_embed
        return offsets

    def vocabSize_all(self):
        """
        功能: 计算总词表大小
        返回:
            - 整数，文本词表大小 + 所有时间序列tokenizer的词表大小总和
        """
        return self.text_vocab_size + sum(tokenizer.n_embed for tokenizer in self.ecgTokenizers)

    def encode(self, input, model_id=1):
        """
        功能: 编码输入(文本或时间序列)为token ID
        输入:
            - input: 
                * 字符串: 文本输入
                * torch.Tensor: 时间序列输入, shape根据数据类型不同而异
            - model_id: 时间序列tokenizer的索引，默认为1
        输出:
            - 编码后的token ID
              * 文本输入: 整数列表
              * 时间序列输入: 整数tensor, shape=[1, seq_len]
        处理流程:
            1. 根据输入类型选择相应的tokenizer
            2. 对文本使用textTokenizer编码
            3. 对时间序列使用对应的ecgTokenizer编码，并加上偏移量
        """
        if isinstance(input, str):
            return self.textTokenizer(input)["input_ids"]
        elif isinstance(input, torch.Tensor):
            input = input.to('cpu')
            if model_id < len(self.ecgTokenizers):
                tokenizer_index = model_id
                _, _, indices = self.ecgTokenizers[tokenizer_index](input)
                return indices + self.offsets[tokenizer_index]
            else:
                raise ValueError(f"Invalid model_id. Please provide a number between 0 and {len(self.ecgTokenizers)}.")
        else:
            raise ValueError("Unsupported input type. Please provide either a string or a torch.Tensor.")
        
    def decode(self, input, skip_special_tokens=True):
        """
        功能: 解码token ID为文本
        输入:
            - input: token ID或ID列表
            - skip_special_tokens: 是否跳过特殊token，默认为True
        输出:
            - 解码后的文本字符串
        说明:
            只解码文本token，时间序列token将被过滤掉
        """
        # 确保input是列表格式
        if isinstance(input, torch.Tensor):
            input = input.tolist()
        elif not isinstance(input, list):
            input = [input]
        
        # 过滤掉时间序列token，只保留文本token
        text_tokens = []
        for token_id in input:
            if token_id < self.text_vocab_size:
                text_tokens.append(token_id)
            # 时间序列token被跳过，不添加到text_tokens中
        
        # 如果没有文本token，返回空字符串
        if not text_tokens:
            return ""
            
        return self.textTokenizer.decode(text_tokens, skip_special_tokens=skip_special_tokens)

# ==================== 新增: Llama 基座版本 ====================

class InstructTimeLlama(LlamaForCausalLM):
    """
    与 InstructTime (GPT-2 版) 相同的多模态逻辑，但基座替换为 Llama。
    仅修改父类、嵌入接口以及 config 中的隐藏维度字段名称。
    """

    def __init__(self, config: LlamaConfig, ecgTokenizers, text_embedding=128256):
        """text_embedding 给个较大的默认值（Llama3 vocab≈128k）"""
        super().__init__(config)

        self.ecgTokenizers = ecgTokenizers

        # ===== 共用时间序列 Embedding 与投影层 =====
        embed_vector = torch.empty(0, self.ecgTokenizers[0].hidden_dim)
        for tokenizer in self.ecgTokenizers:
            tokenizer_embed_vector = copy.deepcopy(tokenizer.quantize.embed).transpose(-1, 0)
            embed_vector = torch.cat([embed_vector, tokenizer_embed_vector], dim=0)
        self.embed_layer = nn.Embedding.from_pretrained(embed_vector)

        self.text_embedding = text_embedding
        # LlamaConfig 的隐藏维度字段为 hidden_size
        self.embed = config.hidden_size

        # pad_token 尚未设置时与 eos_token 保持一致
        self.config.pad_token_id = (
            self.config.eos_token_id if self.config.pad_token_id is None else self.config.pad_token_id
        )

        # domain-wise 投影层
        self.projection_layers = nn.ModuleList()
        for _ in ecgTokenizers:
            mlp = MLP(self.ecgTokenizers[0].hidden_dim, [64, 128, 256, 512], self.embed)
            mlp.apply(self.init_weights_kaiming)
            self.projection_layers.append(mlp)

        # 计算 offsets
        self.offsets = [self.text_embedding]
        for tokenizer in self.ecgTokenizers:
            self.offsets.append(self.offsets[-1] + tokenizer.n_embed)

    @staticmethod
    def init_weights_kaiming(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
            m.bias.data.fill_(0.01)

    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]

        text_mask = torch.lt(input_ids, self.text_embedding)
        ecg_mask = ~text_mask

        # ---------------- 文本 token ----------------
        text_ids = input_ids.clone()
        text_ids[ecg_mask] = self.config.pad_token_id

        # llama 的嵌入层接口如下
        text_embeddings = self.get_input_embeddings()(text_ids)
        text_embeddings.mul_(text_mask.float().unsqueeze(-1))

        # ---------------- 时间序列 token ----------------
        masked_ids = input_ids.clone()
        masked_ids[text_mask] = 0
        masked_ids[ecg_mask] -= self.text_embedding

        ecg_embeddings = torch.zeros_like(text_embeddings)
        for i, _ in enumerate(self.ecgTokenizers):
            tokenizer_mask = (input_ids >= self.offsets[i]) & (input_ids < self.offsets[i + 1])
            tokenizer_ids = input_ids.clone()
            tokenizer_ids[~tokenizer_mask] = 0
            tokenizer_ids[tokenizer_mask] -= self.text_embedding

            tokenizer_embeddings = self.embed_layer(tokenizer_ids)
            tokenizer_embeddings = self.projection_layers[i](tokenizer_embeddings)
            tokenizer_embeddings.mul_(tokenizer_mask.float().unsqueeze(-1))
            ecg_embeddings.add_(tokenizer_embeddings)

        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = ecg_embeddings + text_embeddings

        outputs = super().forward(*args, **kwargs)
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        ecgTokenizers,
        text_embedding=128256,
        device="cpu",
        **kwargs,
    ):
        """加载保存的 InstructTimeLlama 权重（与 GPT2 版逻辑保持一致）"""
        import os
        import json
        # 1. 读取 LlamaConfig
        config = LlamaConfig.from_pretrained(model_path)

        # 2. 构造 MultiTokenizer 以获取词表大小
        multi_tokenizer = MultiTokenizer(ecgTokenizers, base_model="llama3")

        # 3. 初始化模型
        model = cls(config, ecgTokenizers,
                    text_embedding=len(multi_tokenizer.textTokenizer))

        # 4. 扩容 & 替换输出头
        model.resize_token_embeddings(len(multi_tokenizer.textTokenizer))
        current_output = model.get_output_embeddings()
        new_output = nn.Linear(config.hidden_size, multi_tokenizer.vocabSize_all(), bias=False).to(device)
        new_output.weight.data[: len(multi_tokenizer.textTokenizer)] = current_output.weight.data
        model.set_output_embeddings(new_output)
        model.config.vocab_size = multi_tokenizer.vocabSize_all()

        # 5. 加载权重文件（支持分片 safetensors / pytorch）
        state_dict = None
        loaded_path = None
        
        # 首先检查是否有分片的safetensors文件
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            try:
                from safetensors.torch import load_file
                # 加载分片索引文件
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                
                state_dict = {}
                weight_map = index_data.get("weight_map", {})
                
                # 获取所有分片文件
                shard_files = set(weight_map.values())
                
                # 逐个加载分片文件
                for shard_file in shard_files:
                    shard_path = os.path.join(model_path, shard_file)
                    if os.path.exists(shard_path):
                        shard_state = load_file(shard_path, device=device)
                        state_dict.update(shard_state)
                        
                loaded_path = f"分片safetensors文件: {len(shard_files)}个文件"
                print(f"Loading model weights from sharded safetensors: {len(shard_files)} files")
                
            except (ImportError, json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load sharded safetensors: {e}")
                state_dict = None
        
        # 如果分片加载失败，尝试单个文件
        if state_dict is None:
            candidate_paths = [
                os.path.join(model_path, "model.safetensors"),
                os.path.join(model_path, "pytorch_model.bin"),
                os.path.join(model_path, "model.bin"),
            ]

            for p in candidate_paths:
                if os.path.exists(p):
                    loaded_path = p
                    if p.endswith(".safetensors"):
                        try:
                            from safetensors.torch import load_file
                            state_dict = load_file(p, device=device)
                            print(f"Loading model weights from safetensors: {p}")
                        except ImportError:
                            print("Warning: safetensors not installed, trying to load as pytorch format")
                            continue
                    else:
                        state_dict = torch.load(p, map_location=device)
                        print(f"Loading model weights from pytorch format: {p}")
                    break

        if state_dict is None:
            raise FileNotFoundError(f"未找到模型权重文件在 {model_path}")

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys when loading InstructTimeLlama model: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys when loading InstructTimeLlama model: {unexpected_keys}")
            
        print(f"Successfully loaded complete InstructTimeLlama model from {loaded_path}")

        model.to(device)
        return model