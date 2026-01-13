import os
import json
import glob

# 设定图片所在的文件夹路径
IMAGE_DIR = "./raw_data"
OUTPUT_FILE = "metadata.jsonl"


def create_metadata():
    # 支持的图片格式
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    print(f"找到 {len(image_paths)} 张图片...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for img_path in image_paths:
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_path)[0]
            txt_path = base_name + ".txt"

            prompt = ""
            # 如果有对应的 txt 文件，就读取
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    prompt = txt_file.read().strip()
            else:
                # 如果没有，可以用文件名或者空字符串，或者在这里报错
                print(f"警告: {file_name} 没有对应的 txt 文件")
                continue

            # 写入 JSONL 一行
            data = {
                "file_name": file_name,
                "text": prompt
            }
            f.write(json.dumps(data) + "\n")

    print(f"处理完成！数据集索引已保存为 {OUTPUT_FILE}")


if __name__ == "__main__":
    create_metadata()