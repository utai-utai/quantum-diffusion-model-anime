import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class JSONLDataset(Dataset):
    def __init__(self, jsonl_path, image_dir, image_size):
        self.data = []
        self.image_dir = image_dir
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item['file_name'])
        image = Image.open(img_path).convert("RGB")
        text = item['text']
        return self.transform(image), text


def get_dataloader(config):
    dataset = JSONLDataset(
        jsonl_path=config.JSONL_PATH,
        image_dir=config.IMAGE_DIR,
        image_size=config.IMG_SIZE
    )
    return  DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
