import os
import random

import cv2
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from PIL import Image

from ..utils.transforms import crop, generate_target, transform_pixel


class Aligned(Dataset):
    def __init__(self, dataset_info_path: str, img_root: str = ".",
                 input_roi_ratio: float = 0.8,
                 input_size: tuple[int, int] = (256, 256),
                 output_size: tuple[int, int] = (64, 64),
                 target_type: str = "gaussian", sigma: float = 1.5,
                 is_train: bool = True, scale_factor: float = 0.25,
                 rot_factor: float = 30, flip: bool = True):
        self.dataset = pd.read_csv(dataset_info_path)
        self.img_root = img_root
        self.input_roi_ratio = input_roi_ratio
        self.input_size = input_size
        self.output_size = output_size
        self.target_type = target_type
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.flip = flip
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.dataset.iloc[idx, 0])

        img = np.array(Image.open(img_path))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        pts = self.dataset.iloc[idx, 1:].values
        pts = pts.astype(np.float32).reshape(-1, 2)

        h, w = img.shape[:2]
        cx = (w - 1) / 2
        cy = (h - 1) / 2
        scale = max(w * self.input_roi_ratio, h*self.input_roi_ratio) / 200

        nparts = pts.shape[0]

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor)

            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts[:, 0] = w - pts[:, 0]
                cx = w - cx

            if random.random() <= 0.5 and self.flip:
                img = np.flipud(img)
                pts[:, 1] = h - pts[:, 1]
                cy = h - cy

        center = torch.tensor([cx, cy])
        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros(
            (nparts, self.output_size[1], self.output_size[0]), dtype=np.float32)
        tpts = pts.copy()

        for i in range(nparts):
            tpts[i, :] = transform_pixel(
                tpts[i, :], center, scale, self.output_size, rot=r)
            target[i] = generate_target(
                target[i], tpts[i], self.sigma, label_type=self.target_type)

        img = img.astype(np.float32)
        img = (img / 255 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.tensor(target)
        pts = torch.tensor(pts)
        tpts = torch.tensor(tpts)

        meta = {"index": idx, "center": center,
                "scale": scale, "pts": pts, "tpts": tpts}

        return img, target, meta
