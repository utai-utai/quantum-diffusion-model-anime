import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import Config
from src.modules import UNetModel
from src.builder import build_inference_system


@torch.no_grad()
def sample(unet, vae, tokenizer, text_encoder, generated_num, positive_prompts, negative_prompts, scheduler, guidance_scale=7.5):
    device = next(unet.parameters()).device
    # 正向提示词
    positive_input = tokenizer(positive_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    positive_embeddings = text_encoder(positive_input.input_ids.to(device)).last_hidden_state
    # 反向提示词
    negative_input = tokenizer(negative_prompts,padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    negative_embeddings = text_encoder(negative_input.input_ids.to(device)).last_hidden_state
    # 拼接: [positive, negative] -> Shape: [2*B, 77, Dim]
    context = torch.cat([positive_embeddings, negative_embeddings])
    # 初始化噪声
    latents = torch.randn((generated_num, Config.LATENT_DIM, Config.LATENT_SIZE, Config.LATENT_SIZE), device=device)
    print(f"Sampling for {Config.TIMESTEPS} steps on {device}...")

    # DDPM 逆向去噪循环
    for t in tqdm(reversed(range(Config.TIMESTEPS)), total=Config.TIMESTEPS):
        latents_input = torch.cat([latents] * 2)
        t_batch = torch.full((generated_num * 2,), t, device=device, dtype=torch.long)

        # 预测噪声
        noise_pred = unet(latents_input, t_batch, context)
        noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
        noise_pred = noise_pred_negative + guidance_scale * (noise_pred_positive - noise_pred_negative)
        alpha = scheduler.alphas[t].to(device)
        alpha_cumprod = scheduler.alphas_cumprod[t].to(device)
        beta = scheduler.betas[t].to(device)

        # 核心逆向计算
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)
        mean_coeff = 1 / torch.sqrt(alpha)
        pred_mean = mean_coeff * (latents - (beta / sqrt_one_minus_alpha_cumprod) * noise_pred)
        if t > 0:
            noise = torch.randn_like(latents)
            sigma = torch.sqrt(beta)
            latents = pred_mean + sigma * noise
        else:
            latents = pred_mean

    # 解码
    latents = latents / 0.18215
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def inference(generated_num = 10):
    unet = UNetModel(
        in_channels=Config.LATENT_DIM,
        out_channels=Config.LATENT_DIM,
        model_channels=128,
        context_dim=Config.CONTEXT_DIM,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(2, 4),
        n_qubits=Config.N_QUBITS,
        n_layers=Config.N_LAYERS,
        use_quantum=Config.USE_QUANTUM,
    )
    unet, vae, tokenizer, text_encoder, scheduler = build_inference_system(Config, unet)
    weight_path = os.path.join(Config.SAVE_MODEL_PATH, "unet_epoch_30.pth")
    if os.path.exists(weight_path):
        unet.load_state_dict(torch.load(weight_path, map_location=next(unet.parameters()).device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"File not found: {weight_path}")
        return

    # 准备 Prompt
    positive_prompt = "A high quality anime girl portrait, vibrant colors"
    negative_prompt = "low quality, bad anatomy, worst quality, lowres, blurry, watermark, text, error"
    positive_prompts = [positive_prompt] * generated_num
    negative_prompts = [negative_prompt] * generated_num
    images = sample(
        unet=unet,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        generated_num=generated_num,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        scheduler=scheduler,
        guidance_scale=7.5)

    # 保存
    os.makedirs("results/img", exist_ok=True)
    for i, img_arr in enumerate(images):
        img = Image.fromarray(img_arr)
        img_save_path = f"results/img/result_7{i}.png"
        img.save(img_save_path)
        print(f"Saved: {img_save_path}")


if __name__ == "__main__":
    inference()
