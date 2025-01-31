from typing import Callable, Dict, Optional, Union, Tuple, List
from collections import defaultdict
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import monai

# from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker

from ..base import BaseTrainer
from ..models import MultiLoss
from ..metrics import MultiMetric


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Union[MultiLoss, Callable],
                 metric_ftns: Union[MultiMetric, Dict[str, Callable]],
                 optimizer: torch.optim,
                 lr_scheduler: torch.optim.lr_scheduler,
                 config: dict,
                 project: str,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 log_step: int = None,
                 mixed_precision: bool = False,
                 use_monai_inferer: bool = False,
                 inferer_function: Callable = None,
                 ):

        super().__init__(model=model,
                         loss_function=loss_function,
                         metric_ftns=metric_ftns,
                         optimizer=optimizer,
                         config=config,
                         lr_scheduler=lr_scheduler,
                         seed=seed,
                         device=device,
                         project=project,
                         tags=tags,
                         )

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.use_monai_inferer = use_monai_inferer
        self.inferer = monai.inferers.SlidingWindowInferer(
            roi_size=(128, 128, 128),
            sw_batch_size=4,
            overlap=0.25,
            mode="gaussian",
            sigma_scale=0.125,
            )
        self.inferer_function = inferer_function

        self.mixed_precision = mixed_precision

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.inputs_pr_iteration = int(config['inputs_pr_iteration'])

        self.batch_size = data_loader.batch_size
        self.len_epoch = len(data_loader) if not self.iterative else self.inputs_pr_iteration
        self.log_step = int(self.len_epoch/4) if not isinstance(log_step, int) else int(log_step/self.batch_size)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = defaultdict(list)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    out = self.model(data)
                    loss = self._loss(out, target)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(data)
                loss = self._loss(out, target)

                loss.backward()
                self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses['loss'].append(loss)

            if batch_idx % self.log_step == 0:
                self.logger.info('Train {}: {} {} Loss: {:.6f}'.format(
                    'Epoch' if not self.iterative else 'Iteration',
                    epoch,
                    self._progress(batch_idx),
                    loss))

            if batch_idx >= self.inputs_pr_iteration and self.iterative:
                break

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        losses['loss_func'] = str(self.loss_function)

        return {"loss": np.mean(losses["loss"])}

    def _loss(self, out: Union[torch.Tensor, Tuple[torch.Tensor]], target: torch.Tensor):

        if isinstance(out, tuple):
            output, auxiliary = out

            ssum = 1.
            loss = self.loss_function(output, target)
            auxiliary = auxiliary if isinstance(auxiliary, list) else [auxiliary]
            # reverse order
            auxiliary.reverse()
            for t, aux in enumerate(auxiliary, 1):
                loss += self.loss_function(aux, target)/(2.0**t)
                ssum += 1/(2.0**t)
            loss /= ssum
            return loss
        elif isinstance(out, list):
            loss = 0.
            ssum = 0.
            h, w, d = target.shape[-3:]
            for t, o in enumerate(out):
                o_h, o_w, o_d = o.shape[-3:]
                if h != o_h or w != o_w or d != o_d:
                    o = F.interpolate(o, size=(h, w, d), mode='trilinear', align_corners=False)
                loss += self.loss_function(o, target)/(2.0**t)
                ssum += 1/(2.0**t)
            return loss/ssum
        
        output = out

        if len(output.shape) == 6:
            loss = self.loss_function(output[:, 0], target)  # The actual prediction not deep supervision
            ssum = 1.
            for t in range(1, output.shape[1]):
                loss += self.loss_function(output[:, t], target)/(2.0**t)
                ssum += 1/(2.0**t)
            loss /= ssum
            if torch.isnan(loss):
                return torch.tensor(0.0).to(self.device)
            return loss

        loss = self.loss_function(output, target)
        if torch.isnan(loss):
                return torch.tensor(0.0).to(self.device)
        return loss

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.valid_data_loader is None:
            return None

        self.model.eval()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if len(target.shape) < 4 and not self.use_monai_inferer:
                    out = self.model(data)
                else:
                    if self.inferer_function is not None:
                        out = self.inferer(data, self.inferer_function(self.model))
                    else:
                        out = self.inferer(data, self.model)
                loss = self.loss_function(out, target)
                metrics['val_loss'].append(loss.item())

                for key, metric in self.metric_ftns.items():
                    if self.metrics_is_dict:
                        metrics[key].append(metric(out.cpu(), target.cpu()).item())
                    else:
                        metrics[key].append(metric(out, target).item())

        metric_dict = dict()
        for key, item in metrics.items():
            metric_dict[key] = np.mean(metrics[key])

        return metric_dict

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx
            total = self.data_loader.n_samples
        elif hasattr(self.data_loader, 'batch_size'):
            current = batch_idx
            total = self.len_epoch
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
