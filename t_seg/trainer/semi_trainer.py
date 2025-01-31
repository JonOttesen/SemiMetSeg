from typing import Callable, Dict, Union, Tuple, List, Optional
from collections import defaultdict
import math

import numpy as np
import torch
import monai
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from ..base import BaseTrainer


class SemiTrainer(BaseTrainer):
    """
    Trainer class
    """
    METHODS = ['cps', 'ict', 'mt']

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
                 loss_function: callable,
                 metric_ftns: Dict[str, Callable],
                 lr_scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List[torch.optim.lr_scheduler._LRScheduler]],
                 config: dict,
                 project: str,
                 data_loader: torch.utils.data.dataloader.DataLoader,
                 valid_data_loader: torch.utils.data.dataloader.DataLoader,
                 ema_model: Optional[torch.nn.Module] = None,
                 method: Optional[str] = None,
                 semi_supervised: bool = False,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 mixed_precision: bool = True,
                 semi_supervised_loader: Optional[torch.utils.data.dataloader.DataLoader] = None,
                 consistency: Optional[float] = 1.,
                 wait_epoch: int = 0,
                 ):

        super().__init__(
            model=model,
            loss_function=loss_function,
            metric_ftns=metric_ftns,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config,
            seed=seed,
            device=device,
            tags=tags,
            project=project,
            )
            
        self.threshold = config['threshold']
        self.wait_epoch = wait_epoch
        self.wait_factor = wait_epoch // 2

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.semi_supervised_loader = semi_supervised_loader

        self.inferer = monai.inferers.SlidingWindowInferer(
            roi_size=(128, 128, 128),
            sw_batch_size=4,
            overlap=0.25,
            mode="gaussian",
            sigma_scale=0.125,
            )
        
        if method not in self.METHODS and method is not None:
            raise ValueError('Method must be one of {}'.format(self.METHODS))
        self.method = method
        self.semi_supervised = semi_supervised
        
        if self.semi_supervised:
            self.semi_method = getattr(self, self.method)

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        self.batch_size = data_loader.batch_size
        self.inputs_pr_iteration = int(config['inputs_pr_iteration'])
        self.len_epoch = len(data_loader) if not self.iterative else self.inputs_pr_iteration
        self.len_valid = len(valid_data_loader)
        self.mixed_precision = mixed_precision

        self.ema_model = ema_model
        if method in ['mt', 'ict'] and ema_model is None:
            raise ValueError('Must provide an EMA model for MT and ICT')

        if self.method == 'mt' or self.method == 'ict':
            self.ema_model.to(self.device)
            # self.ema_model.eval()
            self.ema_model.requires_grad_(False)
            self.ema_decay = 0.99

        self.sigmoid = lambda x: torch.nn.functional.logsigmoid(x).exp()
        self.kl_distance = nn.KLDivLoss(reduction='none')
        self.consistency = consistency
        self.iter_num = 0
    
    def cps(self, data, semi_data, target, epoch):
        loss, sup_loss_1, sup_loss_2, cps_loss_1, cps_loss_2, consistency_weight = self._cps(data, semi_data, target, epoch)
        loss_dict = {
            'loss': loss,
            'sup_loss_1': sup_loss_1,
            'sup_loss_2': sup_loss_2,
            'cps_loss_1': cps_loss_1,
            'cps_loss_2': cps_loss_2,
            'consistency_weight': torch.Tensor([consistency_weight]),
        }

        return loss_dict

    def ict(self, data, semi_data, target, epoch):
        # ICT mix factors
        ict_alpha = 0.2
        ict_mix_factors = np.random.beta(ict_alpha, ict_alpha, size=(data.shape[0]//2, 1, 1, 1, 1))
        ict_mix_factors = torch.tensor(ict_mix_factors, dtype=torch.float).to(self.device)
        # Continue from here
        batch_size = semi_data.shape[0] // 2
        sup_batch_size = data.shape[0]
        unlabeled_volume_batch_0 = semi_data[0:batch_size, ...]
        unlabeled_volume_batch_1 = semi_data[batch_size:, ...]

        # Mix images
        batch_ux_mixed = unlabeled_volume_batch_0 * (1.0 - ict_mix_factors) + unlabeled_volume_batch_1 * ict_mix_factors
        
        input_volume_batch = torch.cat([data, batch_ux_mixed], dim=0)
        outputs = self.model(input_volume_batch)
        outputs_sig = self.sigmoid(outputs)

        with torch.no_grad():
            ema_output = self.sigmoid(self.ema_model(semi_data))
            batch_pred_mixed = ema_output[0:batch_size, ...] * (1.0 - ict_mix_factors) + ema_output[batch_size:, ...] * ict_mix_factors

        supervised_loss = self._loss(outputs[:sup_batch_size], target)
        consistency_weight = self.get_current_consistency_weight(epoch)
        consistency_loss = torch.mean((outputs_sig[sup_batch_size:] - batch_pred_mixed)**2)
        if epoch < self.wait_factor:
            loss = supervised_loss
        else:
            loss = supervised_loss + consistency_weight * consistency_loss

        return {
            'loss': loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss,
            'consistency_weight': torch.Tensor([consistency_weight]),
            }
    
    def mt(self, data, semi_data, target, epoch):
        noise = torch.clamp(torch.randn_like(semi_data) * 0.1, -0.2, 0.2)
        ema_inputs = semi_data + noise
        sup_batch_size = data.shape[0]

        volume_batch = torch.cat([data, semi_data], dim=0)

        outputs = self.model(volume_batch)
        outputs_soft = self.sigmoid(outputs)

        with torch.no_grad():
            ema_output = self.ema_model(ema_inputs)
            ema_output_sig = self.sigmoid(ema_output)

        supervised_loss = self._loss(outputs[:sup_batch_size], target)
        consistency_weight = self.get_current_consistency_weight(epoch)
        consistency_loss = torch.mean((outputs_soft[sup_batch_size:] - ema_output_sig)**2)

        if epoch < self.wait_factor:
            loss = supervised_loss
        else:
            loss = supervised_loss + consistency_weight * consistency_loss

        return {
            'loss': loss,
            'supervised_loss': supervised_loss,
            'consistency_loss': consistency_loss,
            'consistency_weight': torch.Tensor([consistency_weight]),
            }

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * self.sigmoid_rampup(epoch, self.wait_epoch)

    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def update_ema_variables(self, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        loss_dict = defaultdict(list)

        if self.semi_supervised:
            semi_loader = iter(self.semi_supervised_loader)

        for batch_idx, (data, target) in tqdm(enumerate(self.data_loader), total=self.len_epoch):
            data, target = data.to(self.device), target.to(self.device)

            # No need to fetch data is it si not semi-supervised time
            if self.semi_supervised_loader:
                semi_data = semi_loader.next()
                if not isinstance(semi_data, tuple):
                    semi_data = semi_data.to(self.device)
            
            if isinstance(self.optimizer, list):
                for opt in self.optimizer:
                    opt.zero_grad()
            else:
                self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    if self.semi_supervised:
                        losses = self.semi_method(data, semi_data, target, epoch)
                    else:
                        out = self.model(data)
                        loss = self._loss(out, target)
                        losses = {'loss': loss}
                
                self.scaler.scale(losses['loss']).backward()
    
                if isinstance(self.optimizer, list):
                    for opt in self.optimizer:
                        self.scaler.step(opt)
                else:
                    self.scaler.step(self.optimizer)

                self.scaler.update()
            else:
                if self.semi_supervised:
                    losses = self.semi_method(data, semi_data, target, epoch)
                else:
                    out = self.model(data)
                    loss = self._loss(out, target)
                    losses = {'loss': loss}

                losses['loss'].backward()
                if isinstance(self.optimizer, list):
                    for opt in self.optimizer:
                        opt.step()
                else:
                    self.optimizer.step()
            
            if self.method in ['ict', 'mt']:
                self.update_ema_variables(self.ema_decay, self.iter_num)
                self.iter_num += 1

            for key, item in losses.items():
                loss_dict[key].append(item.item())
            
            if batch_idx >= self.inputs_pr_iteration and self.iterative:
                break
        
        if isinstance(self.optimizer, list):
            for opt in self.optimizer:
                opt.zero_grad()
        else:
            self.optimizer.zero_grad()
            
        if isinstance(self.lr_scheduler, list):
            for lr in self.lr_scheduler:
                lr.step()      
        else:
            self.lr_scheduler.step()

        return {key: np.mean(np.array(value)) for key, value in loss_dict.items()}

    def _loss(self, out: Union[torch.Tensor, Tuple[torch.Tensor]], target: torch.Tensor):
        
        if isinstance(out, (list, tuple)):
            output, auxiliary = out
            
            loss = self.loss_function(output, target)
            auxiliary = auxiliary if isinstance(auxiliary, list) else [auxiliary]
            for aux in auxiliary:
                loss += 0.33*self.loss_function(aux, target)

            if torch.isnan(loss):
                return torch.tensor(0.0).to(self.device)
            return loss
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

    def _cps(self, data, semi_data, target, epoch):
        all_data = torch.cat([data, semi_data], dim=0)

        pred_1 = self.model(all_data, step=1)
        pred_2 = self.model(all_data, step=2)
        sup_loss_1 = self._loss(pred_1[:pred_1.shape[0] // 2], target)
        sup_loss_2 = self._loss(pred_2[:pred_2.shape[0] // 2], target)
        # In case of monai, we have 6 dimensions for deep supervision
        with torch.no_grad():
            if len(pred_1.shape) == 6 and len(pred_2.shape) == 6:
                max_1 = self.sigmoid(pred_1[:, 0]).detach()
                max_2 = self.sigmoid(pred_2[:, 0]).detach()
            else:
                max_1 = self.sigmoid(pred_1).detach()
                max_2 = self.sigmoid(pred_2).detach()

            if self.threshold > 0:
                max_1 = (max_1 > self.threshold).type(torch.float32)
                max_2 = (max_2 > self.threshold).type(torch.float32)

        cps_loss_1 = self._loss(pred_1, max_2)
        cps_loss_2 = self._loss(pred_2, max_1)
        consistency_weight = self.get_current_consistency_weight(epoch)
        if epoch < self.wait_factor:
            loss = sup_loss_1 + sup_loss_2
        else:
            loss = sup_loss_1 + sup_loss_2 + consistency_weight*(cps_loss_1 + cps_loss_2)

        return loss, sup_loss_1, sup_loss_2, cps_loss_1, cps_loss_2, consistency_weight

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        self.model.eval()
        losses = list()
        metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(self.valid_data_loader), total=self.len_valid):
                data, target = data.to(self.device), target.to(self.device)

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