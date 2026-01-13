import torch
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from src.modules import UNetModel
from src.config import Config

# 1. 实例化模型 (建议先用 use_quantum=False 快速测试结构，因为量子模拟很慢)
device = Config.DEVICE
unet = UNetModel(
        in_channels=Config.LATENT_DIM,  # 对应原来的 latent_dim
        out_channels=Config.LATENT_DIM,  # 输出也是 4 通道
        model_channels=64,  # 基础通道数 (显存不够改小，比如 64)
        context_dim=Config.CONTEXT_DIM,  # T5 的维度 (512)
        channel_mult=(1, 2, 4),  # 通道倍数: 128 -> 256 -> 512
        attention_resolutions=(2, 4),  # 在 8x8 和 4x4 的层级使用 Attention
        use_quantum=False,
    ).to(device)

# 2. 创建虚拟输入
# x: [Batch, In_Channels, H, W]
x = torch.randn(Config.BATCH_SIZE, 4, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
# t: [Batch]
t = torch.randint(0, 1000, (Config.BATCH_SIZE,)).to(device)
# context: [Batch, Seq_Len, Context_Dim]
context = torch.randn(Config.BATCH_SIZE, 77, 512).to(device)

# verbose=0 表示不在控制台打印（防止刷屏），只保存到变量里
model_stats = summary(
    unet,
    input_data=[x, t, context],
    col_names=["input_size", "output_size", "num_params"],
    depth=3,
    verbose=0
)

output_path = "unet_structure.txt"
# 注意：一定要用 encoding="utf-8"，因为表格里包含特殊符号 (├──)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(str(model_stats))

print(f"✅ 模型结构已保存到: {output_path}")