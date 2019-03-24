import os
import copy
import cv2
import numpy as np
import torch


colormap = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255]
]

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2image(tensor, one_hot=False):
    array = tensor.detach().cpu().numpy()
    if one_hot:
        c, h, w = array.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
        for k in range(1, c):
            image[array[k] >= 0.5] = colormap[k]
    else:
        array *= 255
        image = np.squeeze(np.transpose(array, (1, 2, 0))).astype(np.uint8)
    return image


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


from .fast_functions import _predict_contour
def find_nuclei(nuclei, contours=None, th=0.5):
    'Expects pred shape (C, H, W), with channel 1 being contours, channels 2-C being nuclei.'
    if nuclei.ndim == 2:
        h, w = nuclei.shape
    elif nuclei.ndim == 3:
        c, h, w = nuclei.shape
    nuclei_segmap = np.zeros((h, w), np.uint8)
    if nuclei.ndim == 2:
        nuclei_segmap[nuclei >= th] = 1
    elif nuclei.ndim == 3:
        for k in range(c):
            nuclei_segmap[nuclei[k] >= th] = 1
    retval, nuclei_instmap = cv2.connectedComponents(nuclei_segmap)
    if not contours is None:
        contours = np.around(contours).astype(np.int32)
        nuclei_instmap = _predict_contour(nuclei_instmap, contours)
        # contours_idx = np.where(contours >= 0.5)
        # contour_map = np.zeros((h, w), np.int32)
        # for i, j in zip(*contours_idx):
        #     l = max(j - 1, 0)
        #     r = min(j + 1, w - 1)
        #     u = max(i - 1, 0)
        #     d = min(i + 1, h - 1)
        #     if pred_map[i, l] > 0:
        #         contour_map[i, j] = pred_map[i, l]
        #     elif pred_map[i, r] > 0:
        #         contour_map[i, j] = pred_map[i, r]
        #     elif pred_map[u, j] > 0:
        #         contour_map[i, j] = pred_map[u, j]
        #     elif pred_map[d, j] > 0:
        #         contour_map[i, j] = pred_map[d, j]
        #     elif pred_map[u, l] > 0:
        #         contour_map[i, j] = pred_map[u, l]
        #     elif pred_map[u, r] > 0:
        #         contour_map[i, j] = pred_map[u, r]
        #     elif pred_map[d, l] > 0:
        #         contour_map[i, j] = pred_map[d, l]
        #     elif pred_map[d, r] > 0:
        #         contour_map[i, j] = pred_map[d, r]
        # pred_map += contour_map

    return nuclei_instmap

