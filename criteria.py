from util.util import find_nuclei
import copy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def check_shape(shape1, shape2):
    try:
        assert(shape1 == shape2)
    except AssertionError:
        print('label shape {} and pred shape {} do not match'.format(label.shape, pred.shape))
        raise


def prf(label, pred, th):
    'Expects labels and preds to be 2d numpy arrays.'
    try:
        assert(label.shape == pred.shape)
    except AssertionError:
        print('label shape {} and pred shape {} do not match.'.format(label.shape, pred.shape))
        raise
    pred[pred >= th] = 1
    pred[pred < th] = 0
    pred = pred.astype(np.uint8)
    precision, recall, f1_score, support = precision_recall_fscore_support(label.flatten(), pred.flatten(), average='binary')
    return {'precision':precision, 'recall':recall, 'f1_score':f1_score}


def dice(label, pred, th=0.5):
    'Expects labels and preds to be 2d numpy arrays.'
    check_shape(label.shape, pred.shape)

    pred = np.copy(pred)
    label = np.copy(label)
    pred[pred < th] = 0
    pred[pred >= th] = 1
    pred = pred.astype(np.uint8)
    label[label < th] = 0
    label[label >= th] = 1
    label = label.astype(np.uint8)
    dice = 2 * np.sum(label[pred == 1]) / (np.sum(label) + np.sum(pred))
    return dice


def multichannel_dice(label, pred, channels, th):
    'Expects label shape (H, W) and pred shape (C, H, W).'
    check_shape(label.shape, pred.shape[1:])

    dices = np.zeros(len(channels))
    for ic, c in enumerate(channels):
        label_c = np.zeros_like(label)
        label_c[label == c] = 1
        dices[ic] = dice(label_c, pred[c], th)
    return dices


def multichannel_prcurve(label, pred, channels=None, ths=None):
    'Expects label shape (H, W) and pred shape (C, H, W).'
    'PR curves are calculated for each channel in `channels`.'
    check_shape(label.shape, pred.shape[1:])
    try:
        assert(label.max() <= pred.shape[0])
    except AssertionError:
        print('label value should be in [0, channel_num]')
        raise
    
    if channels is None:
        channels = np.arange(pred.shape[0])
    if ths is None:
        ths = np.linspace(0, 1, num=20, endpoint=False)
    precision = np.zeros((len(channels), len(ths)))
    recall = np.zeros((len(channels), len(ths)))
    f1 = np.zeros((len(channels), len(ths)))
    
    for ic, c in enumerate(channels):
        label_c = np.zeros_like(label)
        label_c[label == c] = 1
        pred_c = pred[c]
        for ith, th in enumerate(ths):
            bin_pred_c = np.copy(pred_c)
            bin_pred_c[pred_c >= th] = 1
            bin_pred_c[pred_c < th] = 0
            tp = np.sum((label_c == 1) & (bin_pred_c == 1))
            precision[ic, ith] = tp / np.sum(bin_pred_c) if np.sum(bin_pred_c) > 0 else 1
            recall[ic, ith] = tp / np.sum(label_c)
            f1[ic, ith] = 2 * precision[ic, ith] * recall[ic, ith] / (precision[ic, ith] + recall[ic, ith]) if (precision[ic, ith] + recall[ic, ith]) > 0 else 0
    return {'ths': ths, 'precision': precision, 'recall': recall, 'f1': f1}


def iou(label, pred):
    'Expects label shape (H, W) and pred shape (H, W).'
    'Label and pred should contain instance ids.'
    check_shape(label.shape, pred.shape)

    num_true = len(np.unique(label)) - 1
    num_pred = len(np.unique(pred)) - 1
    intersect = np.histogram2d(label.flatten(), pred.flatten(), bins=(num_true + 1, num_pred + 1))[0]
    area_t = np.histogram(label, bins=(num_true + 1))[0][:, np.newaxis]
    area_p = np.histogram(pred, bins=(num_pred + 1))[0][np.newaxis, :]
    union = area_t + area_p - intersect
    iou_ = intersect[1:, 1:] / union[1:, 1:]
    return iou_
   

def ap(label, pred, ths=np.arange(0.5, 1.0, 0.05)):
    'Expects label shape (C, H, W) and pred shape (C, H, W).'
    check_shape(label.shape, pred.shape)
    
    # convert label and pred to be instance maps
    label = find_nuclei(label[2:], contours=label[1])
    pred = find_nuclei(pred[2:], contours=pred[1])

    num_true = len(np.unique(label)) - 1
    num_pred = len(np.unique(pred)) - 1
    iou_ = iou(label, pred)
    ths = ths[np.newaxis, np.newaxis, :]
    matches = iou_[:, :, np.newaxis] > ths

    tp = np.sum(matches, (0, 1))
    fp = num_pred - tp
    fn = num_true - tp
    return tp / (tp + fp + fn + 0.001)


def multiclass_ap(label, pred, ths=np.arange(0.5, 1.0, 0.05)):
    'Expects label shape (C, H, W) and pred shape (C, H, W).'
    check_shape(label.shape, pred.shape)

    c, h, w = label.shape
    cls_ap = []
    for cls in range(2, c):
        cls_label = label[[0, 1, cls]]
        cls_pred = pred[[0, 1, cls]]
        cls_ap.append(ap(cls_label, cls_pred))
    cls_ap = np.array(cls_ap)
    return cls_ap
        
