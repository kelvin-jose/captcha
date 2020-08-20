import torch
import config
import numpy as np
from PIL import Image
from torchvision import transforms


class CaptchaDataset:
    def __init__(self, captchas, labels):
        self.images = captchas
        self.labels = labels
        # self.labels = [name.split('/')[-1][:-4] for name in self.images]
        self.transform = transforms.Compose([
            transforms.Resize((config.height, config.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        captcha = self.images[index]
        image = Image.open(captcha).convert('RGB')
        image = np.array(self.transform(image)).astype('float32')
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'label': torch.tensor(self.labels[index], dtype=torch.long)
        }

