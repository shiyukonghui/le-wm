import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    """
    AdaLN-zero调制函数
    对输入进行缩放和平移变换，常用于自适应层归一化
    
    参数:
        x: 输入张量
        shift: 平移参数
        scale: 缩放参数
    
    返回:
        调制后的张量: x * (1 + scale) + shift
    """
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """
    草图各向同性高斯正则化器 (单GPU版本)
    用于对投影特征进行正则化约束，使其接近各向同性高斯分布
    """

    def __init__(self, knots=17, num_proj=1024):
        """
        初始化SIG正则化器
        
        参数:
            knots: 积分节点数量，用于数值积分
            num_proj: 随机投影的数量
        """
        super().__init__()
        self.num_proj = num_proj
        # 创建时间采样点，范围[0, 3]
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        # 计算梯形积分的步长
        dt = 3 / (knots - 1)
        # 设置积分权重（梯形法则）
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        # 计算高斯窗函数：exp(-t^2/2)，用于加权
        window = torch.exp(-t.square() / 2.0)
        # 注册缓冲区参数（不参与梯度更新）
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        前向传播：计算SIG正则化损失
        
        参数:
            proj: 输入投影张量，形状为 (T, B, D)
                  T: 时间步数, B: 批次大小, D: 特征维度
        
        返回:
            正则化统计量的均值（标量）
        """
        # 生成随机投影矩阵并归一化
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # 计算epps-pulley统计量
        # 将投影后的特征乘以时间点
        x_t = (proj @ A).unsqueeze(-1) * self.t
        # 计算误差：特征函数与理论值的差异
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        # 加权求和得到统计量
        statistic = (err @ self.weights) * proj.size(-2)
        # 返回所有投影和时间步的平均值
        return statistic.mean()


class FeedForward(nn.Module):
    """
    Transformer中使用的前馈神经网络
    包含两个线性层，中间使用GELU激活函数和Dropout
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):
        """
        初始化前馈网络
        
        参数:
            dim: 输入/输出维度
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),           # 层归一化
            nn.Linear(dim, hidden_dim),  # 第一层线性变换
            nn.GELU(),                   # GELU激活函数
            nn.Dropout(dropout),         # Dropout层
            nn.Linear(hidden_dim, dim),  # 第二层线性变换
            nn.Dropout(dropout),         # Dropout层
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (B, T, D)
        
        返回:
            输出张量，形状与输入相同
        """
        return self.net(x)


class Attention(nn.Module):
    """
    带因果掩码的缩放点积注意力机制
    支持多头注意力，常用于自回归模型
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        """
        初始化注意力模块
        
        参数:
            dim: 输入维度
            heads: 注意力头数量
            dim_head: 每个头的维度
            dropout: Dropout概率
        """
        super().__init__()
        inner_dim = dim_head * heads
        # 判断是否需要输出投影（当头数=1且头维度=输入维度时不需要）
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5  # 缩放因子
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        # QKV投影层，一次性生成query、key、value
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # 输出投影层
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        前向传播：计算多头注意力
        
        参数:
            x: 输入张量，形状为 (B, T, D)
               B: 批次大小, T: 序列长度, D: 特征维度
            causal: 是否使用因果掩码（用于自回归模型）
        
        返回:
            注意力输出，形状与输入相同 (B, T, D)
        """
        x = self.norm(x)
        # 训练时使用配置的dropout，推理时不使用
        drop = self.dropout if self.training else 0.0
        # 计算Q、K、V
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, T, inner_dim)
        # 重排张量形状以支持多头注意力
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        # 使用PyTorch内置的高效注意力计算（支持因果掩码）
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        # 重排回原始形状
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """
    带AdaLN-zero条件化的Transformer块
    通过自适应层归一化实现条件控制，常用于扩散模型等生成任务
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        """
        初始化条件化Transformer块
        
        参数:
            dim: 输入维度
            heads: 注意力头数量
            dim_head: 每个注意力头的维度
            mlp_dim: MLP隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        # 层归一化（不带可学习参数，参数由AdaLN生成）
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # AdaLN调制网络：生成6个调制参数（shift和scale各2个，gate各2个）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        # 初始化调制网络最后一层为零，实现zero初始化
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        前向传播：带条件的Transformer块计算
        
        参数:
            x: 输入张量，形状为 (B, T, D)
            c: 条件张量，形状为 (B, T, D)
        
        返回:
            输出张量，形状与输入相同
        """
        # 从条件生成6个调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # 注意力子层：使用AdaLN调制和门控残差连接
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        # MLP子层：使用AdaLN调制和门控残差连接
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """
    标准Transformer块
    包含多头注意力和前馈网络，使用Pre-LN结构
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        """
        初始化标准Transformer块
        
        参数:
            dim: 输入维度
            heads: 注意力头数量
            dim_head: 每个注意力头的维度
            mlp_dim: MLP隐藏层维度
            dropout: Dropout概率
        """
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        # 层归一化（不带可学习参数）
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        """
        前向传播：标准Transformer块计算
        
        参数:
            x: 输入张量，形状为 (B, T, D)
        
        返回:
            输出张量，形状与输入相同
        """
        # 注意力子层：残差连接 + 层归一化
        x = x + self.attn(self.norm1(x))
        # MLP子层：残差连接 + 层归一化
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    标准Transformer模型
    支持AdaLN-zero条件化块或标准块，可用于编码器或解码器
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        """
        初始化Transformer模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
            depth: Transformer层数
            heads: 注意力头数量
            dim_head: 每个注意力头的维度
            mlp_dim: MLP隐藏层维度
            dropout: Dropout概率
            block_class: Transformer块类型（Block或ConditionalBlock）
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        # 输入投影层：将输入维度映射到隐藏维度
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 条件投影层：将条件维度映射到隐藏维度
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        # 输出投影层：将隐藏维度映射到输出维度
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        # 创建指定数量的Transformer块
        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        """
        前向传播：Transformer模型计算
        
        参数:
            x: 输入张量，形状为 (B, T, input_dim)
            c: 条件张量（可选），形状为 (B, T, input_dim)
               仅在使用ConditionalBlock时需要
        
        返回:
            输出张量，形状为 (B, T, output_dim)
        """
        # 输入投影
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        # 条件投影（如果提供了条件）
        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        # 逐层通过Transformer块
        for block in self.layers:
            # 根据块类型选择是否传入条件
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        # 输出投影
        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    """
    嵌入器模块
    将输入特征通过卷积和MLP映射到嵌入空间
    """
    
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        """
        初始化嵌入器
        
        参数:
            input_dim: 输入特征维度
            smoothed_dim: 卷积后的平滑维度
            emb_dim: 最终嵌入维度
            mlp_scale: MLP中间层的缩放因子
        """
        super().__init__()
        # 1D卷积用于特征平滑
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        # MLP嵌入网络
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        前向传播：计算特征嵌入
        
        参数:
            x: 输入张量，形状为 (B, T, D)
               B: 批次大小, T: 序列长度, D: 特征维度
        
        返回:
            嵌入张量，形状为 (B, T, emb_dim)
        """
        x = x.float()
        # 调整维度顺序以适应Conv1d: (B, T, D) -> (B, D, T)
        x = x.permute(0, 2, 1)
        # 通过1D卷积进行特征平滑
        x = self.patch_embed(x)
        # 恢复维度顺序: (B, D, T) -> (B, T, D)
        x = x.permute(0, 2, 1)
        # 通过MLP生成最终嵌入
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """
    简单的多层感知机
    包含可选的归一化层和激活函数
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        """
        初始化MLP
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（默认与输入维度相同）
            norm_fn: 归一化函数类（默认LayerNorm，设为None则不使用）
            act_fn: 激活函数类（默认GELU）
        """
        super().__init__()
        # 如果未指定归一化函数，则使用恒等映射
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (B*T, D)
        
        返回:
            输出张量，形状为 (B*T, output_dim)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """
    自回归预测器
    用于预测下一个时间步的嵌入，基于Transformer架构
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        """
        初始化自回归预测器
        
        参数:
            num_frames: 最大帧数（序列长度）
            depth: Transformer层数
            heads: 注意力头数量
            mlp_dim: MLP隐藏层维度
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（默认与输入维度相同）
            dim_head: 每个注意力头的维度
            dropout: Transformer内部Dropout概率
            emb_dropout: 嵌入层Dropout概率
        """
        super().__init__()
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        # 使用条件化Transformer（支持AdaLN-zero）
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        前向传播：自回归预测
        
        参数:
            x: 输入嵌入张量，形状为 (B, T, d)
               B: 批次大小, T: 序列长度, d: 嵌入维度
            c: 条件张量，形状为 (B, T, act_dim)
               通常为动作或其他条件信息
        
        返回:
            预测的下一帧嵌入，形状为 (B, T, output_dim)
        """
        T = x.size(1)
        # 添加位置编码
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        # 通过条件化Transformer进行预测
        x = self.transformer(x, c)
        return x
