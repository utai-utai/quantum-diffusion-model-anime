import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from src.quantum_circuit import quantum_torch_layer


class SinusoidalPositionEmbeddings(nn.Module):
    """ Diffusion 时间步编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResNetBlock(nn.Module):
    """
    ResNet Block with GroupNorm and Time Embedding injection.
    结构: GroupNorm -> SiLU -> Conv -> + TimeEmb -> GroupNorm -> SiLU -> Conv -> + Input
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=32):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)  # 注入时间信息
        h = h + time_emb[:, :, None, None]  # 广播加法
        h = self.block2(h)
        return h + self.shortcut(x)


class CrossAttention(nn.Module):
    """
    实现 Q, K, V 注意力机制。
    - Self-Attention: context=None, Q=K=V=Image
    - Cross-Attention: context=Text, Q=Image, K=V=Text
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0)  # MVP 暂时不加 Dropout
        )

    def forward(self, x, context=None):
        # x: [Batch, Seq_Len(H*W), Dim]
        h = self.heads
        q = self.to_q(x)
        context = context if context is not None else x  # context: [Batch, Seq_Len_Text, Dim]
        k = self.to_k(context)
        v = self.to_v(context)

        # 拆分多头: (b, seq, heads*dim_head) -> (b, heads, seq, dim_head)
        q = q.view(x.shape[0], -1, h, self.dim_head).permute(0, 2, 1, 3)
        k = k.view(context.shape[0], -1, h, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(context.shape[0], -1, h, self.dim_head).permute(0, 2, 1, 3)

        # Attention Score
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1, is_causal=False)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], -1, h * self.dim_head)
        return self.to_out(out)


class QuantumFeedForward(nn.Module):
    """
    量子前馈层：在初始化时动态构建电路和设备，实现完全解耦。
    """
    def __init__(self, dim, n_qubits, n_layers, q_device_name="default.qubit"):
        super().__init__()
        self.classical_ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.pre_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_qubits)
        )
        self.q_layer = quantum_torch_layer(n_qubits, n_layers, q_device_name)
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim)
        )
        self.gate = nn.Parameter(torch.tensor(0.1))  # 初始权重小一点，保证训练初期稳定

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        out_classical = self.classical_ff(x)
        x_in = torch.atan(self.pre_net(x).to(dtype=torch.float32))   # 降维并归一化，确保输入在 [0, 2pi] 或 [-pi, pi] 之间更有利于量子编码，量子层使用 Float32
        x_q = self.q_layer(x_in.flatten(0, 1)).view(x_in.shape).to(dtype=x.dtype)  # 变成 [Batch*Seq, n_qubits]
        out_quantum = self.post_net(x_q)
        return out_classical + self.gate * out_quantum


class BasicTransformerBlock(nn.Module):
    """
    包含一个 Self-Attn, 一个 Cross-Attn 和一个 FeedForward。
    这是 Stable Diffusion 处理文字生成的核心单元。
    """
    def __init__(self, dim, context_dim, n_qubits, n_layers, heads=8, dim_head=64, q_device = "default.qubit",use_quantum=True):
        super().__init__()
        # 1. Self-Attention (图片看自己，处理空间关系)
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, context_dim=None, heads=heads, dim_head=dim_head)

        # 2. Cross-Attention (图片看文字，处理语义生成)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head)

        # 3. Feed Forward (前馈神经网络)
        self.norm3 = nn.LayerNorm(dim)
        if use_quantum:
            self.ff = QuantumFeedForward(dim, n_qubits, n_layers, q_device)
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )

    def _run_block(self, x, context):
        # x shape: [B, H*W, C]
        x = self.attn1(self.norm1(x)) + x  # Self-Attention
        x = self.attn2(self.norm2(x), context=context) + x  # Cross-Attention
        x = self.ff(self.norm3(x)) + x  # Feed Forward
        return x

    def forward(self, x, context):
        if self.training:
            return checkpoint(self._run_block, x, context, use_reentrant=False)
        else:
            return self._run_block(x, context)


class SpatialTransformer(nn.Module):
    """
    为了把 Transformer 塞进 CNN (UNet) 里，我们需要做维度的变换。
    [B, C, H, W] -> [B, H*W, C] -> Transformer -> [B, C, H, W]
    """
    def __init__(self, channels, context_dim, n_qubits, n_layers, heads=8, dim_head=64, q_device = "default.qubit", use_quantum=True):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.transformer = BasicTransformerBlock(channels, context_dim, n_qubits, n_layers, heads, dim_head, q_device, use_quantum)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_in = x
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)  # Reshape to sequence: [B, C, H, W] -> [B, H*W, C]
        x = self.transformer(x, context)  # Run Transformer
        x = x.permute(0, 2, 1).reshape(b, c, h, w)  # Reshape back: [B, H*W, C] -> [B, C, H, W]
        return self.proj_out(x) + x_in


class UNetModel(nn.Module):
    def __init__(
            self,
            in_channels=4,
            model_channels=128,  # 基础通道数，越大越强 [128, 256, 512, 1024]
            out_channels=4,
            context_dim=512,  # T5-small 的输出维度
            channel_mult=(1, 2, 4),  # 通道倍数，对应 [128, 256, 512]
            attention_resolutions=(2, 4),  # 在哪些层使用 Attention (下采样倍数)
            n_qubits = 2,
            n_layers = 4,
            q_device = "default.qubit",
            use_quantum=True, # 是否使用量子版本
    ):
        super().__init__()

        # 时间嵌入维度
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # === Downsample Path (编码器) ===
        ch = model_channels
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)

        curr_res = 1
        dims = [ch]  # 记录每一层的通道数，给 Skip Connection 用

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            # 每个 Level 加 2 个 ResNet Block
            for _ in range(2):
                layers = [ResNetBlock(ch, out_ch, time_dim)]
                ch = out_ch

                # 判断这一层是否需要 Attention (通常在低分辨率层加，比如 16x16, 8x8)
                if curr_res in attention_resolutions:
                    layers.append(SpatialTransformer(ch, context_dim, n_qubits=n_qubits, n_layers=n_layers, use_quantum=use_quantum))

                self.downs.append(nn.Sequential(*layers))
                dims.append(ch)

            # Downsample (除最后一层外)
            if i != len(channel_mult) - 1:
                self.downs.append(nn.Sequential(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                ))
                curr_res *= 2
                dims.append(ch)

        # === Mid Block (瓶颈层 - 语义理解最深的地方) ===
        self.mid_block1 = ResNetBlock(ch, ch, time_dim)
        self.mid_attn = SpatialTransformer(ch, context_dim, n_qubits=n_qubits, n_layers=n_layers, q_device=q_device, use_quantum=use_quantum)
        self.mid_block2 = ResNetBlock(ch, ch, time_dim)

        # === Upsample Path (解码器) ===
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            for _ in range(3):  # 解码器通常比编码器厚一点
                # Skip Connection: Concatenate
                skip_ch = dims.pop()
                layers = [ResNetBlock(ch + skip_ch, out_ch, time_dim)]
                ch = out_ch

                if curr_res in attention_resolutions:
                    layers.append(SpatialTransformer(ch, context_dim, n_qubits=n_qubits, n_layers=n_layers, use_quantum=use_quantum))

                self.ups.append(nn.Sequential(*layers))

            # Upsample
            if i != 0:
                self.ups.append(nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1)
                ))
                curr_res //= 2

        # === Output ===
        self.norm_out = nn.GroupNorm(32, ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, context):
        # 1. 时间 Embedding
        t_emb = self.time_mlp(t)

        # 2. Input Conv
        x = self.conv_in(x)
        skips = [x]

        # 3. Down Path
        for layer in self.downs:
            if isinstance(layer, nn.Sequential) and isinstance(layer[0], nn.Conv2d) and layer[0].stride[0] == 2:
                # 如果是 Downsample 层
                x = layer(x)
            else:
                # 如果是 ResNet 或 Transformer
                for module in layer:
                    if isinstance(module, ResNetBlock):
                        x = module(x, t_emb)
                    elif isinstance(module, SpatialTransformer):
                        x = module(x, context)
                    else:
                        x = module(x)
            skips.append(x)

        # 4. Middle Path
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x, t_emb)

        # 5. Up Path
        for layer in self.ups:
            if isinstance(layer, nn.Sequential) and isinstance(layer[0], nn.Upsample):
                # 如果是 Upsample 层
                x = layer(x)
            else:
                # Skip Connection Concatenation
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)

                for module in layer:
                    if isinstance(module, ResNetBlock):
                        x = module(x, t_emb)
                    elif isinstance(module, SpatialTransformer):
                        x = module(x, context)
                    else:
                        x = module(x)

        # 6. Output
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        return x
    