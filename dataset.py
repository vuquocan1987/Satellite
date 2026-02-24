"""
Dataset loader for Satellite Flood Imagery — Palette Compatible

Returns dict matching Palette's set_input():
    data['cond_image']  →  (3, H, W)  input March 21 image, normalized [-1, 1]
    data['gt_image']    →  (3, H, W)  target date image, normalized [-1, 1]
    data['path']        →  str        identifier for logging
    data['mask']        →  None       (no inpainting)

Data on disk: 4-channel PNG [R, G, B, inverted_cloud_mask]
We load only RGB (first 3 channels). Cloud mask / snow depth kept as unused metadata.

Data structure:
    data/images/{idx:03d}_{lon}_{lat}/{year}_{month:02d}_{day:02d}.png
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os


# ── constants ──────────────────────────────────────────────────────────────────
INPUT_SUFFIX = "03_21"
TARGET_SUFFIXES = ["04_07", "04_21", "05_07", "05_21", "06_07", "07_07"]


class FloodDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[128, 128], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        ret['gt_image'] = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gt', file_name)))
        ret['cond_image'] = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'cond', file_name)))
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)

def get_dataloader(image_path="data/images", batch_size=4, num_workers=2,
                   image_size=128, shuffle=True, **kwargs):
    dataset = FloodDataset(image_path=image_path, image_size=image_size, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=True)