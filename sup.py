# To fix a monai dataloader bug
import os
from pathlib import Path

import torch
import monai

import numpy as np

from t_seg.dataset.containers import DatasetContainer
from t_seg.dataset.loaders import VolumeLoader

from t_seg.models import MultiLoss

from t_seg.metrics import (
    MultiMetric,
    Accuracy,
    DiceCoefficient
    )

from t_seg.models.losses import (
    WeightedBCEWithLogitsLoss,
    )

from t_seg.trainer import Trainer

from utils import (
    size,
    transforms,
    valid_transforms,
    )
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, required=True)
parser.add_argument('-fold', type=int, required=True, default=0)
parser.add_argument('-amount', type=str, required=True)
parser.add_argument('-folds', type=int, required=False, default=5)


args = parser.parse_args()
batch_size = args.batch_size
amount = str(args.amount)
fold = args.fold
folds = args.folds

if amount not in ["full", "half"]:
    raise ValueError("amount must be half or full")

config = {
    "name": "SupUNet_fold_{}".format(fold),
    "epochs": 1000,
    "iterative": True,
    "images_pr_iteration": 1,
    "val_images_pr_iteration": 1,
    "batch_size": batch_size,
    "inputs_pr_iteration": 250,
    "learning_rate": 2e-4,
    "optimizer": "AdamW",
    "lr_scheduler": "CosineAnnealingLR",
    "save_dir": "/home/MetModels/{}_data/supervised/fold_{}".format(amount, fold),
    "save_period": 250,
    "size": size,
    "weight_decay": 2e-5,
    "fold": fold,
    "folds": folds,
    "amount": amount,
}

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


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

# Make train

data = DatasetContainer().ELITE(
    path="stanford_data_path",
    dataset_type="all",
    source="Stanford",
    dataset_description="Stanford data",
    sequence_statistics=False,
    )

if amount == "half":
    data, _ = data.split(seed=42, split=0.5)

train, valid = data.k_fold(fold=fold, folds=folds)

order_dict = {0: ('bravo', False), 1: ('t1', True), 2: ('t1', False), 3: ('flair', False)}

train_loader = VolumeLoader(
    datasetcontainer=train,
    transforms=transforms,
    sequence_order = order_dict,
    )

valid_loader = VolumeLoader(
    datasetcontainer=valid,
    transforms=valid_transforms,
    sequence_order = order_dict,
    )


names = [Path(entry.segmentation_path).parent.name for entry in valid]

config["files"] = names
config["params"] = params/1e6

loss = [(1, monai.losses.DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, batch=True)), (1, WeightedBCEWithLogitsLoss(weight=10.))]
# loss = monai.losses.DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, batch=True)
loss = MultiLoss(losses=loss)

metrics = {
    'Accuracy': Accuracy(),
    'DiceCoefficient - 0.5': DiceCoefficient(ignore_background=False, treshold=0.5, from_logits=True),
    'DiceCoefficient - 0.1': DiceCoefficient(ignore_background=False, treshold=0.1, from_logits=True),
    'DiceCoefficient - 0.9': DiceCoefficient(ignore_background=False, treshold=0.9, from_logits=True),
    'DiceLoss': monai.losses.DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True),
    'BinaryCrossEntropy': torch.nn.BCEWithLogitsLoss(),
    }

metrics = MultiMetric(metrics=metrics)

sampler = torch.utils.data.RandomSampler(train, replacement=True, 
                                         num_samples=config['inputs_pr_iteration']*config['batch_size'])


train_loader = torch.utils.data.DataLoader(dataset=train_loader,
                                           num_workers=12,
                                           batch_size=config["batch_size"],
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

trainer = Trainer(
    model=model,
    loss_function=loss,
    metric_ftns=metrics,
    optimizer=optimizer,
    config=config,
    data_loader=train_loader,
    valid_data_loader=valid_loader,
    lr_scheduler=lr_scheduler,
    seed=None,
    # log_step=50,
    device="cuda:0",
    mixed_precision=True,
    tags=["{}".format(amount), "3D", "sup", "UNet", "fold_{}".format(fold)],
    project="Supervised",
    )

trainer.train()