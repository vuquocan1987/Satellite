import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs

class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class FloodDataset(data.Dataset):
    """
    Flood satellite dataset with cloud masking.
    
    Returns:
        gt_image:    (3, H, W) target RGB, normalized [-1, 1]
        cond_image:  (3, H, W) condition RGB, normalized [-1, 1]
        cloud_mask:  (1, H, W) valid pixel mask, 1.0 = clear, 0.0 = cloudy
        path:        str
    
    Cloud mask combines BOTH cond and gt masks:
    a pixel is valid only if BOTH images are clear there.
    
    Images on disk are 4-channel RGBA PNGs:
        channels 0-2: RGB
        channel 3: inverted cloud mask (255 = clear, 0 = cloud)
    """
    def __init__(self, data_root, data_flist, data_len=-1, 
                 image_size=[128, 128], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs_rgb = transforms.Compose([
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tfs_mask = transforms.Compose([
            transforms.CenterCrop((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        self.image_size = image_size

    def _load_rgba(self, path):
        """Load RGBA image, return (PIL RGB, PIL mask)."""
        img = Image.open(path).convert('RGBA')
        r, g, b, a = img.split()
        rgb = Image.merge('RGB', (r, g, b))
        mask = a  # channel 3: inverted cloud (255=clear, 0=cloud)
        return rgb, mask

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        cond_rgb, cond_mask = self._load_rgba(
            '{}/{}/{}'.format(self.data_root, 'cond', file_name))
        gt_rgb, gt_mask = self._load_rgba(
            '{}/{}/{}'.format(self.data_root, 'gt', file_name))

        # RGB normalized to [-1, 1]
        ret['cond_image'] = self.tfs_rgb(cond_rgb)
        ret['gt_image'] = self.tfs_rgb(gt_rgb)

        # Cloud masks: [0, 1] where 1 = clear pixel
        cond_m = self.tfs_mask(cond_mask)   # (1, H, W), values in [0, 1]
        gt_m = self.tfs_mask(gt_mask)       # (1, H, W)

        # Combined: pixel is valid only if BOTH are clear
        # Threshold at 0.5 to binarize (handles resize interpolation)
        cloud_mask = ((cond_m > 0.5) & (gt_m > 0.5)).float()  # (1, H, W)
        ret['cloud_mask'] = cloud_mask

        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)
