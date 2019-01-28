import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice(preds, labels):
    'expects labels and preds to be 3d tensors'
    epsilon = 1e-3
    batch_size = preds.size()[0]
    preds = preds.view(batch_size, -1)
    labels = labels.view(batch_size, -1).float()
    intersections = labels * preds
    return 1.0 - torch.mean((2 * torch.sum(intersections, dim=1) + epsilon) / (torch.sum(labels, dim=1) + torch.sum(preds, dim=1) + epsilon))


def cross_entropy_with_soft_dice(preds, labels):
    return F.cross_entropy(preds, labels) * 0.5 + soft_dice(F.softmax(preds, dim=1)[:, 1], labels) * 0.5


def cross_entropy_with_soft_dice_2(preds, labels):
    labels_membrane = torch.zeros_like(labels)
    labels_membrane[labels == 1] = 1
    labels_cytoplasm = torch.zeros_like(labels)
    labels_cytoplasm[labels == 2] = 1
    return F.cross_entropy(preds, labels) * 0.3 \
        + soft_dice(F.softmax(preds, dim=1)[:, 1], labels_membrane) * 0.3 \
        + soft_dice(F.softmax(preds, dim=1)[:, 2], labels_cytoplasm) * 0.3


class GANLoss(nn.Module):
    'expects pred and label to be of one example'
    
    def __init__(self, device, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        self.device = device

    def __call__(self, pred, target_is_real):
        if target_is_real:
            label = self.real_label
        else:
            label = self.fake_label
        label = label.expand_as(pred)
        label = label.to(self.device)
        return self.loss(pred, label)
