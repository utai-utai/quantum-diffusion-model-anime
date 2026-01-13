import torch


class Config:
    # =======================================================================
    # --- 参数 ---
    # =======================================================================

    # 数据参数
    IMG_SIZE = 128  # 输入图片大小
    BATCH_SIZE = 64  # 显存小就调小这个
    EPOCHS = 30  # 训练轮数
    WARMUP_EPOCHS = 1
    NUM_WORKERS = 4

    # 学习率
    LR = 1e-4  # 学习率
    LR_QUANTUM = LR * 5.0  # 5e-4 (让量子层学快点)
    LR_GATE = LR * 0.1  # 1e-5 (让Gate稳住别乱动)

    # 模型参数
    LATENT_DIM = 4  # VAE 压缩后的通道数
    LATENT_SIZE = IMG_SIZE // 8  # VAE 缩小 8 倍
    CONTEXT_DIM = 512  # T5-small 的输出维度
    USE_CHECKPOINTING = True

    # 量子参数
    USE_QUANTUM = True
    N_QUBITS = 4
    N_LAYERS = 2
    Q_DEVICE = "default.qubit"

    # 扩散参数
    TIMESTEPS = 1000  # 扩散步数
    START_BETA = 0.0001
    END_BETA = 0.02
    TEXT_DROP_PROB = 0.1

    # =======================================================================
    # --- 路径 ---
    # =======================================================================

    # 预训练模型路径
    VAE_ID = "stabilityai/sd-vae-ft-mse"
    T5_ID = "google-t5/t5-small"

    # 数据集路径
    JSONL_PATH = "dataset/metadata.jsonl"  # json 路径
    IMAGE_DIR = "dataset/raw_data"  # 图片路径
    # JSONL_PATH = "dataset/metadata_mini.jsonl"  # json 路径
    # IMAGE_DIR = "dataset/mini_test"  # 图片路径

    # 保存路径
    SAVE_MODEL_PATH = "results/weight"  # 模型权重保存地址
    SAVE_IMAGE_PATH = "results/img"  # 采样结果保存地址
    # RESUME_PATH = "./saved_models/checkpoint-5"
    RESUME_PATH = None