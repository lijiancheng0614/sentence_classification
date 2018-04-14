from copy import deepcopy

import torch
import math


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        x = torch.FloatTensor(predictions)
        y = torch.FloatTensor(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x, y))

    def mse(self, predictions, labels):
        x = torch.FloatTensor(predictions)
        y = torch.FloatTensor(labels)
        return torch.mean((x - y) ** 2)

    def spearman(self, predictions, labels):
        x = torch.FloatTensor(predictions)
        y = torch.FloatTensor(labels)
        x = x - x.mean()
        y = y - y.mean()
        return torch.dot(x,y) / math.sqrt(torch.dot(x ** 2, y ** 2))

