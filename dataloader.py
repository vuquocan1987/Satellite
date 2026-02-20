from PIL import Image
import torchvision.transforms as T
import torch
import pandas as pd
import os

input_img_suffix = "03_21"
output_img_suffix = ["04_07", "04_21", "05_07", "05_21", "06_07", "07_07", "07_21"]

class FloodDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, image_path = "data/images",transform=None):
        coords = []
        for folder in os.listdir(image_path):
            _, lon, lat = folder.split("_")
            coords.append((float(lon), float(lat)))
        
        
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Assuming masks are grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask