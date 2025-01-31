from optparse import Option
import time
import sys

from typing import List, Callable, Union, Dict, Optional
from abc import abstractmethod
from pathlib import Path
from datetime import datetime

import torch
import py3nvml
import wandb

import numpy as np

from ..logger import get_logger
from ..models import MultiLoss
from ..metrics import MultiMetric


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: MultiLoss,
                 metric_ftns: Union[MultiMetric, Dict[str, callable]],
                 optimizer: torch.optim,
                 config: dict,
                 project: str,
                 lr_scheduler: torch.optim.lr_scheduler,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 ):
        """
        Args:
            model (torch.nn.Module): The model to be trained
            loss_function (MultiLoss): The loss function or loss function class
            metric_ftns (MultiMetric, Dict[str, callable]): Dict or Multimetric for the metrics to be evaluated during validation
            optimizer (torch.optim): torch.optim, i.e., the optimizer class
            config (dict): dict of configs
            project (str): project in wandb
            lr_scheduler (torch.optim.lr_scheduler): pytorch lr_scheduler for manipulating the learning rate
            seed (int): integer seed to enforce non stochasticity,
            device (str): string of the device to be trained on, e.g., "cuda:0"
            tags (List(str)): tags for wandb 
        """

        # Reproducibility is a good thing
        if isinstance(seed, int):
            torch.manual_seed(seed)

        self.config = config

        self.run = wandb.init(
            config=config,
            entity='',
            project=project,
            tags=tags, 
            save_code=False, 
            reinit=True, 
            name=config['name'], 
            mode="online",
            )
        
        self.logger = get_logger(name=__name__)

        # setup GPU device if available, move model into configured device
        if device is None:
            self.device, device_ids = self.prepare_device(config['n_gpu'])
        else:
            self.device = torch.device(device)
            device_ids = list()

        self.model = model.to(self.device)
        self.lr_scheduler = lr_scheduler

        # TODO: Use DistributedDataParallel instead
        if len(device_ids) > 1 and config['n_gpu'] > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        if isinstance(loss_function, dict):
            for key, item in loss_function.items():
                loss_function[key] = item.to(self.device)
            self.loss_function = loss_function
        else:
            self.loss_function = loss_function.to(self.device)

        if isinstance(metric_ftns, dict):  # dicts can't be sent to the gpu
            self.metrics_is_dict = True
            self.metric_ftns = metric_ftns
        else:  # MetricTracker class can be sent to the gpu
            self.metrics_is_dict = False
            self.metric_ftns = metric_ftns.to(self.device)

        self.optimizer = optimizer

        self.epochs = config['epochs']
        self.save_period = config['save_period']

        self.iterative = bool(config['iterative'])

        self.start_epoch = 1

        self.checkpoint_dir = Path(config['save_dir']) / Path(datetime.today().strftime('%Y-%m-%d'))

        self.min_validation_loss = sys.float_info.max  # Minimum validation loss achieved, starting with the larges possible number

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        Args:
            epoch (int): Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validation logic after an epoch
        Args:
            epoch (int): Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        # Use iterations or epochs
        epochs = self.epochs

        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start_time = time.time()
            
            loss_dict = self._train_epoch(epoch)
            val_dict = self._valid_epoch(epoch)
            
            epoch_end_time = time.time() - epoch_start_time

            self.logger.info('Epoch/iteration {} with validation completed in {}.'.format(epoch, epoch_end_time))

            if hasattr(self.lr_scheduler, 'get_last_lr'):
                current_lr = self.lr_scheduler.get_last_lr()[0]
            elif hasattr(self.lr_scheduler, 'get_lr'):
                current_lr = self.lr_scheduler.get_lr()[0]

            if val_dict is not None:
                loss_val_dict = {
                    **loss_dict, 
                    **val_dict, 
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    }
            else:
                loss_val_dict = {
                    **loss_dict, 
                    "epoch": epoch,
                    "learning_rate": current_lr,
                    }
                
            wandb.log(loss_val_dict, commit=True)

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch, best=False)
            
            if val_dict is not None:
                val_loss = val_dict["val_loss"]
                if val_loss < self.min_validation_loss:
                    self.min_validation_loss = val_loss
                    self.save_checkpoint(epoch, best=True)

            self.logger.info('-----------------------------------')
        self.save_checkpoint(epoch, best=False)

    def prepare_device(self, n_gpu_use: int):
        """
        setup GPU device if available, move model into configured device
        Args:
            n_gpu_use (int): Number of GPU's to use
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = n_gpu
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu

        free_gpus = py3nvml.get_free_gpus()

        list_ids = [i for i in range(n_gpu) if free_gpus[i]]
        n_gpu_use = min(n_gpu_use, len(list_ids))

        device = torch.device('cuda:{}'.format(list_ids[0]) if n_gpu_use > 0 else 'cpu')
        if device.type == 'cpu':
            self.logger.warning('current selected device is the cpu, you sure about this?')

        self.logger.info('Selected training device is: {}:{}'.format(device.type, device.index))
        self.logger.info('The available gpu devices are: {}'.format(list_ids))

        return device, list_ids

    def save_checkpoint(self, epoch, best: bool = False):
        """
        Saving checkpoints at the given moment
        Args:
            epoch (int), the current epoch of the training
            bool (bool), save as best epoch so far, different naming convention
        """
        arch = type(self.model).__name__
        if self.lr_scheduler is not None:  # In case of None
            scheduler_state_dict = self.lr_scheduler.state_dict()
        else:
            scheduler_state_dict = None

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            # 'optimizer': self.optimizer.state_dict(),
            # 'scheduler': scheduler_state_dict,
            'config': self.config,
            'loss_func': str(self.loss_function),
            }

        if best:  # Save best case with different naming convention
            save_path = Path(self.checkpoint_dir) / Path('best_validation')
            filename = str(save_path / 'checkpoint-best.pth')
        else:
            save_path = Path(self.checkpoint_dir) / Path('epoch_' + str(epoch))
            filename = str(save_path / 'checkpoint-epoch{}.pth'.format(epoch))

        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def resume_checkpoint(self,
                          resume_model: Union[str, Path],
                          wandb_path: str
                          ):
        """
        Resume from saved checkpoints
        Args:
            resume_model (str, pathlib.Path): Checkpoint path, either absolute or relative
            resume_metric (str, pathlib.Path): Metric path, either absolute or relative
        """
        if not isinstance(resume_model, (str, Path)):
            self.logger.warning('resume_model is not str or Path object but of type {}, '
                                'aborting previous checkpoint loading'.format(type(resume_model)))
            return None

        if not Path(resume_model).is_file():
            self.logger.warning('resume_model object does not exist, ensure that {} is correct, '
                                'aborting previous checkpoint loading'.format(str(resume_model)))
            return None

        resume_model = str(resume_model)
        self.logger.info("Loading checkpoint: {} ...".format(resume_model))

        try:
            checkpoint = torch.load(resume_model, map_location='cpu')
        except:
            checkpoint = torch.load(resume_model)
        self.start_epoch = checkpoint['epoch'] + 1


        self.model.load_state_dict(checkpoint['state_dict'])

        self.model = self.model.to(self.device)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

        self.checkpoint_dir = Path(resume_model).parent.parent  # Ensuring the same main folder after resuming

        api = wandb.Api()
        a = api.run("{}/{0}".format(wandb_path, self.resume_id)).scan_history()
        val_loss = [row["val_loss"] for row in a]
        self.min_validation_loss = min(np.min(np.array(val_loss)), self.min_validation_loss)