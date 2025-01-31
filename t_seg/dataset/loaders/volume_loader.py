from typing import Dict, Optional, Tuple
import random
import time
import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import monai
import numpy as np
from tqdm import tqdm

import monai

from ..containers import DatasetContainer

from ...logger import get_logger

@contextlib.contextmanager
def temp_seed(seed):
    """
    Source:
    https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)

class VolumeLoader(torch.utils.data.Dataset):
    """
    An iterable datasetloader for the dataset container to make my life easier
    """

    def __init__(self,
                 datasetcontainer: DatasetContainer,
                 transforms: monai.transforms.Compose = None,
                 channels: int = 4,
                 sequence_order: Dict[int, Tuple[str, bool]] = None,
                 semi: bool = False,
                 one_hot: Optional[int] = None,
                 ):
        """
        Args:
            datasetcontainer: The datasetcontainer that is to be loaded
            transforms: Transforms the data is gone through before model input
            channels: number of channels for model input
            sequence_order: order of the sequences used for the model
            semi: drop gt or not
            one_hot: do one hot or not?
        """

        self.datasetcontainer = datasetcontainer
        self.transforms = transforms
        self.channels = channels
        self.sequence_order = sequence_order
        self.semi = semi
        self.one_hot = one_hot

        if self.semi:
            self.indicies = self._get_indicies_with_all_modalities()

        self.logger = get_logger(name=__name__)
    
    def _get_indicies_with_all_modalities(self):
        indicies = []
        self.datasetcontainer.order()
        for i, entry in enumerate(self.datasetcontainer):
            sequences = list(deepcopy(self.sequence_order).values())
            for instance in entry:
                contrast = instance.contrast
                sequence_type = instance.sequence_type.lower()

                if (sequence_type, contrast) in sequences:
                    # delete element with (sequence_type, contrast) from sequences
                    if (sequence_type, contrast) in sequences:
                        sequences.remove((sequence_type, contrast))

            if len(sequences) <= len(self.sequence_order.keys()) - self.channels:
                indicies.append(i)
        return indicies
    
    def get_timeserie_points(self):
        subjects = dict()
        indicies_in_subjects = list()
        for i in self.indicies:
            entry = self.datasetcontainer[i]
            name = Path(entry[0].image_path).parts[-3]
            if name not in subjects.keys():
                subjects[name] = 0
            
            subjects[name] += 1
            indicies_in_subjects.append(name)
        
        weights = list()
        for i in indicies_in_subjects:
            weights.append(subjects[i])
        return weights
    
    def load_data(self, index):
        # Fast, not a bottleneck
        if self.semi:
            index = self.indicies[index]
        entry = self.datasetcontainer[index]

        images = [0]*self.channels

        # This is fast now
        if not self.semi:
            gt = entry.open().get_fdata()
            gt = torch.from_numpy(gt).unsqueeze(0).to(torch.float32)

        # Fast enough
        # Order the sequences
        overlapping_channels = list()
        if self.sequence_order is not None:
            for i, instance in enumerate(entry):
                for item, order in self.sequence_order.items():
                    if str(order).lower() == str((instance.sequence_type.lower(), instance.contrast)).lower():
                        # if you have multiple similar sequences, only take one. However, dicts can't have multiple keys so this is a workaround
                        if round(item) in overlapping_channels:
                            if np.random.rand() > 0.5:
                                continue
                            else:
                                img = instance.open()
                                img_seq = img.get_fdata()
                                images[round(item)] = img_seq
                        else:
                            img = instance.open()
                            img_seq = img.get_fdata()
                            images[round(item)] = img_seq
                            overlapping_channels.append(item)
        else:
            images = list()
            for i, instance in enumerate(entry):
                img = instance.open()
                img_seq = img.get_fdata()
                images.append(img_seq)
            
            if len(images) > self.channels:
                # randomly select channels
                random.shuffle(images)
                images = images[:self.channels]

        if self.channels > 1:
            images = np.stack(images, axis=0)
        else:
            images = np.array(images)

        voxels = (img.header["pixdim"][1], img.header["pixdim"][2], img.header["pixdim"][3])
        if self.semi:
            out = {'image': torch.from_numpy(images).to(torch.float32), 'voxel_sizes': voxels}
        else:
            out = {'image': torch.from_numpy(images).to(torch.float32), 'mask': gt, 'voxel_sizes': voxels}
        return out

    def __len__(self):
        if self.semi:
            return len(self.indicies)
        return len(self.datasetcontainer)

    def __getitem__(self, index):
        out = self.load_data(index)
        # start = time.time()
        z = self.transforms(out)
        if isinstance(z, list):
            z = z[0]
        # z['mask'][:] = 1.
        # print(f"Time to transform data: {time.time() - start}")
        if self.semi:
            return z['image']
        mask = torch.Tensor(z['mask'])
        if self.one_hot is not None:
            mask = torch.nn.functional.one_hot(mask[0].to(torch.int64), num_classes=self.one_hot).permute(3, 0, 1, 2).to(torch.float32)
        return torch.Tensor(z['image']), mask

    def __iter__(self):
        self.current_index = 0
        self.max_length = len(self)
        return self

    def __next__(self):
        if not self.current_index < self.max_length:
            raise StopIteration
        item = self[self.current_index]
        self.current_index += 1
        return item
