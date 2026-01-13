import torch.optim as optim
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from src.utils import load_models


def build_training_system(config, dataloader, unet):
    # 1. 初始化 Accelerator (最先做，因为它决定了设备)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        project_dir=config.SAVE_MODEL_PATH
    )
    print(f"--- Building System on {accelerator.device} ---")

    # 2. 加载模型
    unet, vae, tokenizer, text_encoder, noise_scheduler, ema = load_models(config, unet)
    unet.train()

    # 3. 配置优化器
    quantum_params = []
    gate_params = []
    classical_params = []
    for name, param in unet.named_parameters():
        if not param.requires_grad:
            continue
        if 'gate' in name:
            gate_params.append(param)  # A. 识别 Gate 参数 (关键：它是控制开关)
        elif 'q_layer' in name or 'pre_net' in name or 'post_net' in name:
            quantum_params.append(param)  # B. 识别量子相关参数 (QuantumFeedForward)
        else:
            classical_params.append(param)  # C. 其他都是经典参数 (Conv, Linear, Norm 等)
    optimizer = optim.AdamW([
        {'params': classical_params, 'lr': config.LR, 'weight_decay': 1e-2},
        {'params': quantum_params, 'lr': config.LR_QUANTUM, 'weight_decay': 1e-2},
        {'params': gate_params, 'lr': config.LR_GATE * 0.1,'weight_decay': 0.0}
    ])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=config.EPOCHS * len(dataloader)
    )

    # 4. Accelerate Prepare，只有需要训练更新的对象才放进去
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    # 5. 处理未被 Prepare 的对象设备归属
    noise_scheduler.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    ema.to(accelerator.device)

    # 8. 打包返回所有训练循环需要的东西
    return (
        accelerator,
        unet,
        optimizer,
        dataloader,
        lr_scheduler,
        vae,
        tokenizer,
        text_encoder,
        noise_scheduler,
        ema
    )


def build_inference_system(config, unet):
    accelerator = Accelerator(mixed_precision="fp16")  # 初始化 Accelerator
    unet, vae, tokenizer, text_encoder, noise_scheduler, ema = load_models(config, unet)  # 调用模型
    unet.eval()  # 切换到评估模式
    noise_scheduler.to(accelerator.device)
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    return (
        unet,
        vae,
        tokenizer,
        text_encoder,
        noise_scheduler
    )