import os

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.spike_layer import SpikeConv, LIFAct, tdBatchNorm2d, SpikePool


def split_weights(net):
    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

        else:
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)

    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]