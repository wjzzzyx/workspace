import numpy as np
from sklearn.metrics import precision_recall_fscore_support

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


def dice(label, pred, th):
	'Expects labels and preds to be 2d numpy arrays.'
	try:
		assert(label.shape == pred.shape)
	except AssertionError:
		print('label shape {} and pred shape {} do not match'.format(label.shape, pred.shape))
		raise
	pred[pred >= th] = 1
	pred[pred < th] = 0
	pred = pred.astype(np.uint8)
	dice = 2 * np.sum(label[pred == 1]) / (np.sum(label) + np.sum(pred))
	return dice


def multichannel_dice(label, pred, channels, th):
	'Expects label shape (H, W) and pred shape (C, H, W).'
	dices = np.zeros(len(channels))
	for ic, c in enumerate(channels):
		label_c = np.zeros_like(label)
		label_c[label == c] = 1
		dices[ic] = dice(label_c, pred[c], th)
	return dices


def multichannel_prcurve(label, pred, channels=None, ths=None):
	'Expects label shape (H, W) and pred shape (C, H, W).'
	'PR curves are calculated for each channel in `channels`.'
	try:
		assert(label.shape == pred.shape[1:])
	except AssertionError:
		print('label shape {} and pred shape {} do not match'.format(label.shape, pred.shape))
		raise
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
