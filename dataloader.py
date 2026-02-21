# dataloader.py

from PIL import Image
import torchvision.transforms as T
import torch
import xarray as xr
import numpy as np
import glob
import os
from torch.utils.data import DataLoader

# ── constants ──────────────────────────────────────────────────────────────────
INPUT_SUFFIX    = "03_21"
OUTPUT_SUFFIXES = ["04_07", "04_21", "05_07", "05_21", "06_07", "07_07", "07_21"]
YEARS           = list(range(2016, 2026))


# ── FloodDataset ───────────────────────────────────────────────────────────────
class FloodDataset(torch.utils.data.Dataset):
    """
    Each sample:
        input_img  (3, H, W)  — RGB only, march-21 image
        target     (4, H, W)  — RGB + cloud mask, single future date
        snow       scalar     — snow depth on input date from ERA5
        coord      (2,)       — [lon, lat_normalized]
        meta       dict       — lon, lat, year, suffix
    """

    def __init__(self, image_path="data/images", snow_path="data/snow",
                 years=YEARS, transform=None):
        self.transform = transform
        self.samples   = []

        files = sorted(glob.glob(os.path.join(snow_path, "era5_land_daily_*.nc")))
        self.snow = xr.open_mfdataset(files, combine="by_coords")["sd"]

        for folder in os.listdir(image_path):
            _, lon, lat = folder.split("_")
            lon, lat    = float(lon), float(lat)

            for year in years:
                input_path = os.path.join(image_path, folder, f"{year}_{INPUT_SUFFIX}.png")
                if not os.path.exists(input_path):
                    continue

                for suffix in OUTPUT_SUFFIXES:
                    p_clean  = os.path.join(image_path, folder, f"{year}_{suffix}.png")
                    p_cloudy = os.path.join(image_path, folder, f"{year}_{suffix}_C.png")
                    if os.path.exists(p_clean):
                        target_path = p_clean
                    elif os.path.exists(p_cloudy):
                        target_path = p_cloudy
                    else:
                        continue

                    self.samples.append((input_path, target_path, lon, lat, year, suffix))

        lats         = [s[3] for s in self.samples]
        self.lat_min = min(lats)
        self.lat_max = max(lats)

    def _get_snow(self, lon, lat, year, month, day):
        val = self.snow.sel(
            valid_time=np.datetime64(f"{year}-{month:02d}-{day:02d}"),
            latitude=lat, longitude=lon, method="nearest"
        ).values
        return float(val) if not np.isnan(val) else 0.0

    def _normalize_lat(self, lat):
        return (lat - self.lat_min) / (self.lat_max - self.lat_min + 1e-6)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path, lon, lat, year, suffix = self.samples[idx]

        input_img = Image.open(input_path).convert("RGB")
        if self.transform:
            input_img = self.transform(input_img)

        target = Image.open(target_path).convert("RGBA")
        if self.transform:
            target = self.transform(target)

        month, day = map(int, INPUT_SUFFIX.split("_"))
        snow = torch.tensor(
            self._get_snow(lon, lat, year, month, day), dtype=torch.float32)

        coord = torch.tensor([lon, self._normalize_lat(lat)], dtype=torch.float32)

        meta = {"lon": lon, "lat": lat, "year": year, "suffix": suffix}
        return input_img, target, snow, coord, meta


# ── get_loader ─────────────────────────────────────────────────────────────────
def get_loader(image_path="data/images", snow_path="data/snow", years=YEARS,
               batch_size=16, img_size=128, num_workers=4, shuffle=True):

    transform = T.Compose([
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])

    dataset = FloodDataset(
        image_path=image_path,
        snow_path=snow_path,
        years=years,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset ready: {len(dataset)} samples | "
          f"{len(set(s[2] for s in dataset.samples))} locations | "
          f"lat range [{dataset.lat_min:.4f}, {dataset.lat_max:.4f}] | "
          f"years {min(s[4] for s in dataset.samples)}–{max(s[4] for s in dataset.samples)}")
    return loader


# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = get_loader(batch_size=4, snow_path = "data/Meteorological")
    inp, target, snow, coord, meta = next(iter(loader))
    print("input :", inp.shape)      # (4, 3, 128, 128)
    print("target:", target.shape)   # (4, 4, 128, 128)
    print("snow  :", snow.shape)     # (4,)
    print("coord :", coord.shape)    # (4, 2)
    print("meta  :", meta)

