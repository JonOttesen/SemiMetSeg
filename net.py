# encoding: utf-8
from typing import Callable, Dict, Union, Tuple, List, Optional

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet


class SegNetworkV2(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 deep_supr_num: int,
                 ):
        super(SegNetworkV2, self).__init__()

        self.branch1 = DynUNet(spatial_dims=3,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        filters=(48, 96, 128, 192, 256, 384, 512),
                        dropout=0,
                        norm_name='INSTANCE',
                        act_name='leakyrelu',
                        deep_supervision=True,
                        deep_supr_num=deep_supr_num,
                        res_block=False,
                        trans_bias=False)

        self.branch2 = DynUNet(spatial_dims=3,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                        filters=(48, 96, 128, 192, 256, 384, 512),
                        dropout=0,
                        norm_name='INSTANCE',
                        act_name='leakyrelu',
                        deep_supervision=True,
                        deep_supr_num=deep_supr_num,
                        res_block=False,
                        trans_bias=False)

    def forward(self, data, step=1):
        if not self.training and step is None:
            pred1 = self.branch1(data)
            pred2 = self.branch2(data)
            return (pred1, pred2)

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)