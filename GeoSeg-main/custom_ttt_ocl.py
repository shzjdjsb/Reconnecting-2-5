import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from tools.ocl import output_contrastive_loss
from train_supervision import Supervision_Train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=Path, required=True)
    parser.add_argument('-o', '--output_path', type=Path, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--network', choices=['ema', 'net'], default='net')
    parser.add_argument('--tta', choices=['lr', 'd4'], default='d4')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--sample_count', type=int, default=2048)
    parser.add_argument('--downsample', type=int, default=8)
    parser.add_argument('--positive_weight', type=float, default=3.0)
    parser.add_argument('--negative_weight', type=float, default=1.0)
    return parser.parse_args()


def select_network(model, network_name):
    network = model.ema_net if network_name == 'ema' else model.net
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)

    trainable = []
    for parameter in network.decoder.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)
    if not trainable:
        raise RuntimeError('No trainable decoder parameters were found.')
    return network, trainable


def adapt_network(network, optimizer, images, args):
    network.eval()
    loss_value = None
    for _ in range(args.steps):
        prediction = network(images)
        flipped_images = torch.flip(images, dims=[3])
        flipped_prediction = torch.flip(network(flipped_images), dims=[3])
        loss = output_contrastive_loss(
            prediction,
            flipped_prediction,
            positive_weight=args.positive_weight,
            negative_weight=args.negative_weight,
            downsample=args.downsample,
            sample_count=args.sample_count,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    return loss_value


def build_predictor(network, tta_name):
    if tta_name == 'lr':
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
    else:
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Rotate90(angles=[90]),
            tta.Scale(
                scales=[0.5, 0.75, 1.0, 1.25, 1.5],
                interpolation='bicubic',
                align_corners=False,
            ),
        ])
    return tta.SegmentationTTAWrapper(network, transforms)


def main():
    args = get_args()
    if args.steps < 1:
        raise ValueError('--steps must be at least 1 for OCL adaptation.')

    config = py2cfg(args.config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = Supervision_Train.load_from_checkpoint(
        f'{config.weights_path}/{config.test_weights_name}.ckpt', config=config
    ).to(device)
    network, trainable = select_network(checkpoint, args.network)
    optimizer = torch.optim.SGD(
        trainable, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )
    predictor = build_predictor(network, args.tta)

    args.output_path.mkdir(exist_ok=True, parents=True)
    loader = DataLoader(
        config.test_dataset, batch_size=args.batch_size, num_workers=4,
        pin_memory=True, shuffle=False
    )

    for batch in tqdm(loader, desc='ocl predict'):
        images = batch['img'].to(device, non_blocking=True)
        with torch.enable_grad():
            adapt_network(network, optimizer, images, args)
        with torch.no_grad():
            prediction = predictor(images)
            if isinstance(prediction, (tuple, list)):
                prediction = prediction[0]
            masks = prediction.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for mask, image_id in zip(masks, batch['img_id']):
            cv2.imwrite(str(args.output_path / f'{image_id}.png'), mask)


if __name__ == '__main__':
    main()
