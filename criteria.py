import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def prf(label, pred, th):
	'Expects labels and preds to be 2d numpy arrays.'
	try:
		assert(label.shape == pred.shape)
	except AssertionError:
		print('label shape {} and pred shape {} do not match.'.format(label.shape, pred.shape))
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
