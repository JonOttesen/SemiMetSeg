import os
from pathlib import Path

import torch
import monai
import torch.nn as nn

import numpy as np

from tqdm import tqdm

from t_seg.dataset.containers import DatasetContainer
from t_seg.dataset.loaders import VolumeLoader

from t_seg.models import MultiLoss

from t_seg.models.losses import (
    WeightedBCEWithLogitsLoss,
    )

from t_seg.metrics import (
    MultiMetric,
    Accuracy,
    DiceCoefficient
    )

from t_seg.trainer import SemiTrainer

from utils import (
    size,
    transforms,
    semi_transforms,
    valid_transforms,
    )

from net import SegNetworkV2

import warnings
warnings.filterwarnings("ignore")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-fold', type=int, required=True, default=0)
parser.add_argument('-folds', type=int, required=False, default=5)
parser.add_argument('-amount', type=str, required=True)

parser.add_argument('-consistency', type=float, required=True)
parser.add_argument('-method', type=str, required=True)
parser.add_argument('-threshold', type=float, required=False, default=0.5)

args = parser.parse_args()
fold = args.fold
folds = args.folds
amount = str(args.amount)
consistency = args.consistency
method = args.method
threshold = float(args.threshold)

if amount not in ["full", "half"]:
    raise ValueError("amount must be half or full")

config = {
    "name": f"semi_{method}_{consistency}_{fold}",
    "epochs": 1000,
    "iterative": True,
    "batch_size": 4,
    "inputs_pr_iteration": 250,
    "learning_rate": 2e-4,
    "optimizer": "AdamW",
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": "/home/MetModels/{}_data/semi/{}/fold_{}/consistency_{}/threshold_{}".format(amount, method, fold, consistency, threshold),
    "save_period": 250,
    "size": size,
    "wait_epoch": 100,
    "threshold": threshold,
    "weight_decay": 2e-5,
    "fold": fold,
    "folds": folds,
    "method": method,
    "consistency": consistency,
}

# kernels, strides = get_kernels_strides(size, voxel_sizes)

if method == "cps":
    model = SegNetworkV2(
        in_channels=4,
        out_channels=1,
        deep_supr_num=2,
        )
else:
    model = monai.networks.nets.DynUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        filters=(48, 96, 128, 192, 256, 384, 512),
        dropout=0,
        norm_name='INSTANCE',
        act_name='leakyrelu',
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
        trans_bias=False)

ema_model = None
if method in ["ict", "mt"]:
    ema_model = monai.networks.nets.DynUNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=1,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        filters=(48, 96, 128, 192, 256, 384, 512),
        dropout=0,
        norm_name='INSTANCE',
        act_name='leakyrelu',
        deep_supervision=True,
        deep_supr_num=2,
        res_block=False,
        trans_bias=False)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

config["params"] = params / 1e6

# Make train

data = DatasetContainer().ELITE(
    path="Stanford data",
    dataset_type="all",
    source="Stanford",
    dataset_description="Data from Stanford",
    sequence_statistics=False,
    )

if amount == "half":
    data, _ = data.split(seed=42, split=0.5)

train, valid = data.k_fold(fold=fold, folds=folds)

semi = DatasetContainer().ELITE(
    path="Semi Stanford Data",
    dataset_type="all",
    source="Stanford",
    dataset_description="Semi Stanford Data",
    sequence_statistics=False,
    )

for entry in semi:
    for instance in entry:
        if instance.sequence_type.lower() == "flair":
            instance.contrast = False

order_dict = {0: ('bravo', False), 1: ('t1', True), 2: ('t1', False), 3: ('flair', False)}
order_dict_semi = {0: ('bravo', True), 1: ('t1', True), 2: ('t1', False), 3: ('flair', False)}  # Bravo is actually post contrast


train_loader = VolumeLoader(
    datasetcontainer=train,
    transforms=transforms,
    sequence_order = order_dict,
    semi=False
    )

semi_loader = VolumeLoader(
    datasetcontainer=semi,
    transforms=semi_transforms,
    sequence_order=order_dict_semi,
    semi=True,
    )

weights = semi_loader.get_timeserie_points()
weights = 1/np.log(np.float64(np.array(weights))*np.e)


valid_loader = VolumeLoader(
    datasetcontainer=valid,
    transforms=valid_transforms,
    sequence_order = order_dict,
    )


names = [Path(entry.segmentation_path).parent.name for entry in valid]
config["files"] = names
config["params"] = params/1e6

loss = [(1, monai.losses.DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, batch=True)), (1, WeightedBCEWithLogitsLoss(weight=10.))]
# loss = [(1, monai.losses.DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, batch=True))]

loss = MultiLoss(losses=loss)
# loss = monai.losses.DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, batch=True)

metrics = {
    'Accuracy': Accuracy(),
    'DiceCoefficient - 0.5': DiceCoefficient(ignore_background=False, treshold=0.5, from_logits=True),
    'DiceCoefficient - 0.1': DiceCoefficient(ignore_background=False, treshold=0.1, from_logits=True),
    'DiceCoefficient - 0.9': DiceCoefficient(ignore_background=False, treshold=0.9, from_logits=True),
    'DiceLoss': monai.losses.DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True),
    'BinaryCrossEntropy': torch.nn.BCEWithLogitsLoss(),
    }

metrics = MultiMetric(metrics=metrics)

train_sampler = torch.utils.data.RandomSampler(train_loader, replacement=True, 
                                                num_samples=config['inputs_pr_iteration']*config['batch_size'] // 2)

train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=6,
                                           batch_size=config["batch_size"] // 2,
                                           sampler=train_sampler,
                                           )

factor = 2 if config['method'] == 'ict' else 1
sampler = torch.utils.data.WeightedRandomSampler(
    weights=weights,
    replacement=True,
    generator=None,
    num_samples=factor*config['inputs_pr_iteration']*config['batch_size'] // 2,
    )

semi_loader = torch.utils.data.DataLoader(dataset=semi_loader,
                                           num_workers=6,
                                           batch_size=factor*config["batch_size"] // 2,
                                           sampler=sampler,
                                           )

valid_loader = torch.utils.data.DataLoader(dataset=valid_loader,
                                           num_workers=4,
                                           batch_size=1,
                                           shuffle=False,
                                           )


class LRPolicy(object):
    def __init__(self, initial, warmup_steps=10):
        self.warmup_steps = warmup_steps
        self.initial = initial

    def __call__(self, step):
        return self.initial + step/self.warmup_steps*(1 - self.initial)

warmup_steps = 50


optimizer = torch.optim.AdamW(params=model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer,  LRPolicy(initial=1e-2, warmup_steps=warmup_steps))
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(warmup_steps - config["epochs"]))

lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])



trainer = SemiTrainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    optimizer=optimizer,
    config=config,
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    semi_supervised_loader=semi_loader,
    lr_scheduler=lr_scheduler,
    ema_model=ema_model,
    seed=None,
    device="cuda:0",
    mixed_precision=True,
    tags=["3D", "{}".format(method), "UNet", "fold_{}".format(fold), "consistency_{}".format(consistency), "{}".format(amount)],
    project="Semi Supervised",
    wait_epoch=config["wait_epoch"],
    method=config["method"],
    consistency=config["consistency"],
    semi_supervised=True,
    )

trainer.train()