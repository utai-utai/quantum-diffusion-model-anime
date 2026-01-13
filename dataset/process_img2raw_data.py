import os
import random
from pathlib import Path
from tqdm import tqdm  # 进度条库

# ================= 配置区域 =================
# 图片所在的文件夹名称
SOURCE_FOLDER = Path("anime")

# 想要存放txt文件的目标文件夹名称
TARGET_FOLDER = Path("raw_data")

# 随机Prompt列表
TAG_OPTIONS = [
    # 选项1：标准高质量
    "masterpiece, best quality, anime portrait, solo, 8k resolution, highly detailed",
    # 选项2：柔和梦幻光影
    "masterpiece, best quality, anime portrait, soft lighting, cinematic lighting, dreamy atmosphere, depth of field",
    # 选项3：鲜艳明亮风格
    "masterpiece, best quality, anime portrait, vibrant colors, vivid, sharp focus, illustration, aesthetic",
    # 选项4：复古/胶片感
    "masterpiece, best quality, anime portrait, retro style, lo-fi, film grain, nostalgic, warm lighting",
    # 选项5：特写与细节
    "masterpiece, best quality, anime portrait, face focus, extreme detailed eyes, macro photography style",
]


# ===========================================

def generate_random_tags():
    # 1. 检查源文件夹是否存在
    if not SOURCE_FOLDER.exists():
        print(f"错误: 找不到文件夹 '{SOURCE_FOLDER}'，请确认路径。")
        return

    # 2. 如果目标文件夹不存在，自动创建
    if not TARGET_FOLDER.exists():
        print(f"文件夹 '{TARGET_FOLDER}' 不存在，正在创建...")
        TARGET_FOLDER.mkdir(parents=True, exist_ok=True)

    # 3. 获取所有 jpg 文件 (也可以加上 .png)
    # glob('*.jpg') 会找到所有 jpg 结尾的文件
    image_files = list(SOURCE_FOLDER.glob('*.jpg'))

    total_files = len(image_files)
    print(f"检测到 {total_files} 张 JPG 图片，准备开始生成 Tag...")

    # 4. 遍历图片并生成对应的 txt
    # tqdm 用于显示进度条
    for img_path in tqdm(image_files, desc="Processing"):
        # 获取文件名（不带后缀），例如 '0001.jpg' -> '0001'
        file_stem = img_path.stem

        # 拼接目标 txt 路径: raw_data/0001.txt
        txt_path = TARGET_FOLDER / f"{file_stem}.txt"

        # 随机选择一个 Tag
        selected_tag = random.choice(TAG_OPTIONS)

        # 写入文件 (使用 utf-8 编码防止乱码)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(selected_tag)

    print(f"\n成功！已在 '{TARGET_FOLDER}' 中生成了 {total_files} 个 txt 文件。")


if __name__ == "__main__":
    generate_random_tags()