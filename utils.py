from __future__ import division
from __future__ import print_function

import math

import torch

# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes)
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0][floor-1] = 1
    else:
        target[0][floor-1] = ceil - label
        target[0][ceil-1] = label - floor
    return target
