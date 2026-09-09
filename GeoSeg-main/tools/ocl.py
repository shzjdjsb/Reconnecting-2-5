"""Output Contrastive Loss for test-time adaptation."""

import torch
import torch.nn.functional as F


def _main_logits(prediction):
    if isinstance(prediction, (tuple, list)):
        return prediction[0]
    return prediction


def _flatten_pixels(values):
    return values.permute(0, 2, 3, 1).reshape(values.shape[0], -1, values.shape[1])


def _sample_aligned_pixels(values, augmented_values, sample_count):
    values = _flatten_pixels(values)
    augmented_values = _flatten_pixels(augmented_values)
    if sample_count is not None and values.shape[1] > sample_count:
        indices = torch.randperm(values.shape[1], device=values.device)[:sample_count]
        values = values[:, indices]
        augmented_values = augmented_values[:, indices]
    return values, augmented_values


def output_contrastive_loss(
    prediction,
    augmented_prediction,
    positive_weight=3.0,
    negative_weight=1.0,
    downsample=8,
    sample_count=2048,
):
    """Compute OCL between aligned segmentation predictions.

    ``prediction`` and ``augmented_prediction`` are logits for the same image
    pixels under two aligned views, with shape ``[B, C, H, W]``. The positive
    term aligns the same pixel across views; the negative term separates
    different pixels within each view.
    """
    prediction = _main_logits(prediction)
    augmented_prediction = _main_logits(augmented_prediction)
    if prediction.shape != augmented_prediction.shape:
        raise ValueError(
            'prediction and augmented_prediction must have the same shape, '
            f'got {tuple(prediction.shape)} and {tuple(augmented_prediction.shape)}'
        )

    prediction = F.softmax(prediction, dim=1)
    augmented_prediction = F.softmax(augmented_prediction, dim=1)
    if downsample > 1:
        prediction = F.avg_pool2d(prediction, downsample, downsample)
        augmented_prediction = F.avg_pool2d(
            augmented_prediction, downsample, downsample
        )
    prediction, augmented_prediction = _sample_aligned_pixels(
        prediction, augmented_prediction, sample_count
    )

    if prediction.shape[1] < 2:
        return prediction.sum() * 0.0

    prediction = F.normalize(prediction, dim=2)
    augmented_prediction = F.normalize(augmented_prediction, dim=2)

    positive = -(prediction * augmented_prediction).sum(dim=2).mean()

    indices = torch.randperm(prediction.shape[1], device=prediction.device)
    valid = indices != torch.arange(prediction.shape[1], device=prediction.device)
    negative_prediction = (
        prediction[:, valid] * prediction[:, indices[valid]]
    ).sum(dim=2).mean()
    negative_augmented = (
        augmented_prediction[:, valid] * augmented_prediction[:, indices[valid]]
    ).sum(dim=2).mean()
    negative = 0.5 * (negative_prediction + negative_augmented)

    return positive_weight * positive + negative_weight * negative
