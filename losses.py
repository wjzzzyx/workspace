import numpy as np
import torch
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
