import torch
from diffusers import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from copy import deepcopy


def load_models(config, unet):
    # 1. 噪声调度器
    noise_scheduler = NoiseScheduler(
        start_beta=config.START_BETA,
        end_beta=config.END_BETA,
        timesteps=config.TIMESTEPS,
        device="cpu",  # 暂时放 CPU，后面谁用谁移动
    )

    # 2. 加载预训练模型 (VAE, T5)
    print(f"Loading VAE: {config.VAE_ID}...")
    vae = AutoencoderKL.from_pretrained(config.VAE_ID)
    vae.eval()
    vae.requires_grad_(False)
    print(f"Loading T5: {config.T5_ID}...")
    tokenizer = T5Tokenizer.from_pretrained(config.T5_ID, legacy=False)
    text_encoder = T5EncoderModel.from_pretrained(config.T5_ID)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # 3. 构建 UNet
    if config.USE_CHECKPOINTING:
        unet.use_checkpointing = True
    return unet, vae, tokenizer, text_encoder, noise_scheduler, EMA(unet)


class NoiseScheduler:
    """前向扩散过程"""
    def __init__(self, start_beta, end_beta, timesteps, device="cpu"):
        self.device = device
        self.betas = torch.linspace(start_beta, end_beta, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        """手动将内部所有 Tensor 移动到指定设备"""
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(self, original_samples, noise, timesteps):
        """formula: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon"""
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - self.alphas_cumprod[timesteps])
        # 调整形状以便广播 [B] -> [B, 1, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = deepcopy(model)  # 影子模型
        self.model.eval()
        self.decay = decay
        for param in self.model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def to(self, device):
        self.model.to(device)
        return self
