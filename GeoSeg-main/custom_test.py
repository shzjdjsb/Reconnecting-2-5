import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from train_supervision import Supervision_Train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=Path, required=True)
    parser.add_argument('-o', '--output_path', type=Path, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    parser.add_argument('-t', '--tta', default=None, choices=['lr', 'd4'])
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Supervision_Train.load_from_checkpoint(
        f'{config.weights_path}/{config.test_weights_name}.ckpt', config=config
    ).to(device).eval()
    if args.tta == 'lr':
        model = tta.SegmentationTTAWrapper(
            model, tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        )
    elif args.tta == 'd4':
        model = tta.SegmentationTTAWrapper(
            model,
            tta.Compose([
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(
                    scales=[0.5, 0.75, 1.0, 1.25, 1.5],
                    interpolation='bicubic',
                    align_corners=False,
                ),
            ]),
        )
    args.output_path.mkdir(exist_ok=True, parents=True)
    loader = DataLoader(
        config.test_dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(loader, desc='predict'):
            prediction = model(batch['img'].to(device))
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            masks = prediction.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for mask, image_id in zip(masks, batch['img_id']):
                cv2.imwrite(str(args.output_path / f'{image_id}.png'), mask)


if __name__ == '__main__':
    main()
