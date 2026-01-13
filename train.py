import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import shutil

from dataset.dataloader import get_dataloader
from src.config import Config
from src.modules import UNetModel
from src.builder import build_training_system


def train():
    # 1. 准备数据和模型
    dataloader = get_dataloader(Config)
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

    # 构建训练系统
    accelerator, unet, optimizer, dataloader, lr_scheduler, vae, tokenizer, text_encoder, noise_scheduler, ema = build_training_system(
        Config, dataloader, unet)

    gate_history = []
    start_epoch = 0  # 默认为 0

    # ================= [新增] 断点续训加载逻辑 =================
    if Config.RESUME_PATH and os.path.exists(Config.RESUME_PATH):
        print(f"Resuming training from: {Config.RESUME_PATH}")

        # 1. 恢复 Accelerator 状态 (模型权重, 优化器, LR, 随机种子)
        accelerator.load_state(Config.RESUME_PATH)

        # 2. 恢复 EMA 状态 (需要手动处理，因为 EMA 通常不被 accelerator 自动管理)
        ema_path = os.path.join(Config.RESUME_PATH, "ema_state.pth")
        if os.path.exists(ema_path):
            ema.model.load_state_dict(torch.load(ema_path, map_location=accelerator.device))
            print("EMA state loaded.")

        # 3. 计算从第几个 Epoch 开始
        # 假设保存路径格式为 ".../checkpoint-epoch_X"
        try:
            # 尝试从文件夹名字解析 epoch，例如 "checkpoint-10" -> 10
            folder_name = os.path.basename(os.path.normpath(Config.RESUME_PATH))
            last_epoch = int(folder_name.split('-')[-1])
            start_epoch = last_epoch + 1
            print(f"Starting from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not parse epoch from path, defaulting to config or 0. Error: {e}")

        # 4. [关键] 恢复 Gate 参数的冻结/解冻状态
        # 如果跳过了 warmup 阶段，必须确保 gate 参数是可训练的
        if start_epoch > Config.WARMUP_EPOCHS:
            for name, param in unet.named_parameters():
                if 'gate' in name:
                    param.requires_grad = True
            print("Gate parameters unfrozen (Resumed after warmup).")
    # ========================================================

    # 循环从 start_epoch 开始
    for epoch in range(start_epoch, Config.EPOCHS):
        # Warmup 逻辑保持不变，但因为 range 变了，这里会自动处理
        if epoch < Config.WARMUP_EPOCHS:  # warm up
            for name, param in unet.named_parameters():
                if 'gate' in name:
                    param.requires_grad = False
                    param.data.fill_(0.2)
        elif epoch == Config.WARMUP_EPOCHS:
            # 刚好到达解冻的那一轮
            for name, param in unet.named_parameters():
                if 'gate' in name:
                    param.requires_grad = True
        else:
            pass

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}",
                    disable=not accelerator.is_local_main_process)

        for step, (images, texts) in enumerate(pbar):
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215
                if random.random() < Config.TEXT_DROP_PROB:
                    texts = [""] * len(texts)
                tokens = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77).to(
                    accelerator.device)
                context = text_encoder(tokens.input_ids).last_hidden_state

            t = torch.randint(0, Config.TIMESTEPS, (latents.shape[0],), device=accelerator.device)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, t)

            noise_pred = unet(noisy_latents, t, context)
            loss = nn.MSELoss()(noise_pred, noise)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            ema.update(accelerator.unwrap_model(unet))

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if step % 100 == 0 and accelerator.is_local_main_process:
                current_gates = [p.item() for n, p in unet.named_parameters() if 'gate' in n]
                if current_gates:
                    avg = sum(current_gates) / len(current_gates)
                    gate_history.append(avg)
                    # print(f"Gate History: {gate_history}")

        # ================= [新增] 完整的保存逻辑 =================
        if accelerator.is_main_process:
            # 1. 保存用于推理的权重 (User原本的逻辑，保持不变方便测试)
            save_path = os.path.join(Config.SAVE_MODEL_PATH, f"unet_epoch_{epoch + 1}.pth")
            torch.save(ema.model.state_dict(), save_path)

            # 2. 保存用于恢复训练的完整 Checkpoint
            # 建议每隔几个 epoch 保存一次，或者覆盖保存，防止硬盘爆满
            checkpoint_dir = os.path.join(Config.SAVE_MODEL_PATH, f"checkpoint-{epoch}")
            accelerator.save_state(checkpoint_dir)

            # 单独把 EMA 的状态存进去 (accelerator 通常只存 model/optimizer)
            torch.save(ema.model.state_dict(), os.path.join(checkpoint_dir, "ema_state.pth"))

            # (可选) 清理旧的 checkpoint 以节省空间
            # if epoch > 0:
            #     old_ckpt = os.path.join(Config.SAVE_MODEL_PATH, f"checkpoint-{epoch-1}")
            #     if os.path.exists(old_ckpt):
            #         shutil.rmtree(old_ckpt)
        # ========================================================

        # 确保多卡训练时保存操作完成后再继续
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()