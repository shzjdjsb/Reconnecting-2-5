import os
import random
from pathlib import Path

import albumentations as albu
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transform import Compose, RandomScale, SmartCropV1


# Replace these display names with the competition's official class names when known.
CLASSES = tuple(f'class_{i}' for i in range(9))
# Competition masks use raw ID 0 for invalid/ignored pixels.  GeoSeg's
# losses and SmartCrop conventionally use 255 as the ignore sentinel, so
# convert only this custom competition dataset at the dataset boundary.
RAW_IGNORE_INDEX = 0
IGNORE_INDEX = 255
PALETTE = [
    [255, 255, 255], [255, 0, 0], [255, 255, 0],
    [0, 0, 255], [0, 255, 0], [255, 128, 0],
    [128, 0, 255], [0, 200, 200], [128, 128, 128],
]


def _convert_competition_mask(mask):
    """Map competition Ignore ID 0 to GeoSeg's internal sentinel 255."""
    arr = np.array(mask, dtype=np.uint8, copy=True)
    arr[arr == RAW_IGNORE_INDEX] = IGNORE_INDEX
    return Image.fromarray(arr, mode='L')


def _normalize(image, mask):
    crop = Compose([
        RandomScale(
            scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'
        ),
        SmartCropV1(
            crop_size=512, max_ratio=0.75, ignore_index=IGNORE_INDEX, nopad=False
        ),
    ])
    image, mask = crop(image, _convert_competition_mask(mask))
    result = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.RandomBrightnessContrast(
            brightness_limit=0.25, contrast_limit=0.25, p=0.25
        ),
        albu.Normalize(),
    ])(image=np.asarray(image), mask=np.asarray(mask))
    return result['image'], result['mask']


def _validate(image, mask):
    result = albu.Normalize()(image=np.asarray(image), mask=np.asarray(mask))
    return result['image'], result['mask']


class CustomRemoteSensingDataset(Dataset):
    """PNG image/mask pairs with a deterministic train/validation split."""

    def __init__(self, data_root='/home/tqg/train', split='train', val_ratio=0.1,
                 seed=42, transform=None):
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / 'images'
        self.mask_dir = self.data_root / 'masks'
        self.split = split
        self.transform = transform

        image_ids = {p.stem for p in self.image_dir.glob('*.png')}
        mask_ids = {p.stem for p in self.mask_dir.glob('*.png')}
        if image_ids != mask_ids:
            missing_masks = sorted(image_ids - mask_ids)[:5]
            missing_images = sorted(mask_ids - image_ids)[:5]
            raise RuntimeError(
                f'Image/mask mismatch: missing_masks={missing_masks}, '
                f'missing_images={missing_images}'
            )

        ids = sorted(image_ids)
        rng = random.Random(seed)
        rng.shuffle(ids)
        val_count = max(1, int(len(ids) * val_ratio))
        self.img_ids = ids[val_count:] if split == 'train' else ids[:val_count]
        if not self.img_ids:
            raise RuntimeError(f'No samples found for split={split}')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = Image.open(self.image_dir / f'{img_id}.png').convert('RGB')
        mask = _convert_competition_mask(
            Image.open(self.mask_dir / f'{img_id}.png').convert('L')
        )
        if image.size != mask.size:
            raise RuntimeError(f'Size mismatch for {img_id}: {image.size} vs {mask.size}')

        image, mask = self.transform(image, mask)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        mask = torch.from_numpy(np.array(mask, dtype=np.int64, copy=True)).long()
        return {'img': image, 'gt_semantic_seg': mask, 'img_id': img_id}


class CustomRemoteSensingTestDataset(Dataset):
    """Unlabeled 1024x1024 PNG images for competition inference."""

    def __init__(self, data_root='/home/tqg/test_1', transform=None):
        self.image_dir = Path(data_root) / 'images'
        self.transform = transform or _validate
        self.img_ids = sorted(p.stem for p in self.image_dir.glob('*.png'))
        if not self.img_ids:
            raise RuntimeError(f'No PNG test images found in {self.image_dir}')

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = Image.open(self.image_dir / f'{img_id}.png').convert('RGB')
        image, _ = self.transform(image, np.zeros(image.size[::-1], dtype=np.uint8))
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
        return {'img': image, 'img_id': img_id}
