import os

import torch
from torch.utils.data import DataLoader

from geoseg.datasets.custom_dataset import (
    CLASSES,
    CustomRemoteSensingDataset,
    CustomRemoteSensingTestDataset,
    _normalize,
    _validate,
)
from geoseg.losses import UnetFormerLoss
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead, process_model_params


max_epoch = 100
ignore_index = 255
# The competition's raw mask uses ID 0 for Ignore.  The dataset converts it
# to 255 before loss/crop processing, and metrics exclude the unused class-0
# output channel from mIoU/F1 while also masking 255 pixels.
metric_ignore_index = 255
metric_exclude_classes = (0,)
train_batch_size = 16
val_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = 'unetformer-r18-custom-512-crop-ms-e100'
weights_path = f'model_weights/custom/{weights_name}'
test_weights_name = weights_name
log_name = f'custom/{weights_name}'
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None
gpus = 'auto'
resume_ckpt_path = None

# Use pretrained weights only when they are already cached or the server has access
# to the model host. The default keeps training usable on offline servers.
use_pretrained = os.environ.get('GEOSEG_PRETRAINED', '0') == '1'
net = UNetFormer(num_classes=num_classes, pretrained=use_pretrained)
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = True

data_root = os.environ.get('GEOSEG_DATA_ROOT', '/home/tqg/train')
train_dataset = CustomRemoteSensingDataset(
    data_root=data_root, split='train', transform=_normalize
)
val_dataset = CustomRemoteSensingDataset(
    data_root=data_root, split='val', transform=_validate
)
test_dataset = CustomRemoteSensingTestDataset(
    data_root=os.environ.get('GEOSEG_TEST_ROOT', '/home/tqg/test_1'),
    transform=_validate,
)

train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=4,
    pin_memory=True, shuffle=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=val_batch_size, num_workers=4,
    pin_memory=True, shuffle=False, drop_last=False
)

layerwise_params = {
    'backbone.*': dict(lr=backbone_lr, weight_decay=backbone_weight_decay)
}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max_epoch, eta_min=1e-6
)
