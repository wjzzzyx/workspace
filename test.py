import os
import sys
import argparse
from skimage import io
import numpy as np
import torch
import torch.nn.functional as F


def get_args():
	parser = argparse.ArgumentParser()
	# model parameters
	parser.add_argument('--in-channels', type=int, default=1, help='number of input channels.')
	parser.add_argument('--out-channels', type=int, default=2, help='number of output channels.')
	# storage parameters
	parser.add_argument('--data-path', type=str, default='/mnt/ccvl15/yixiao/kaggle/data', help='data path.')
	parser.add_argument('--model-file', type=str, default='/mnt/ccvl15/yixiao/kaggle/models/snapshots/Unet_final.pt', help='path of pretrained model and snapshots.')
	parser.add_argument('--result-path', type=str, default='/mnt/ccvl15/yixiao/kaggle/results', help='path of prediction maps and analysis files.')
	return parser.parse_args()


def test_model(model, dataloader, criterion):
	model.eval()
	model.to(device)
	print('Begin testing...')

	score = 0.0
	channels = np.array([1, 2])
	ths = np.linspace(0, 1, 20, endpoint=False)
	precision = np.zeros((len(channels), len(ths)))
	recall = np.zeros((len(channels), len(ths)))
	f1 = np.zeros((len(channels), len(ths)))
	
	for icase, (image, label) in enumerate(dataloader):
		image = image.to(device)
		with torch.no_grad():
			pred = model(image)
			pred = F.softmax(pred, dim=1)
		pred = pred[0].cpu().numpy()
		label = label[0].numpy()
		# running_score = criterion(label, pred)
		# score += running_score
		res = criterion(label, pred, channels, ths)
		precision += res['precision']
		recall += res['recall']
		f1 += res['f1']

		# pred[pred >= 0.5] = 1
		# pred[pred < 0.5] = 0
		# pred = (pred * 255).astype(np.uint8)
		# io.imsave(os.path.join(args.result_path, str(icase) + '.png'), pred)
		# print('Testing image {}, dice {:.4f}.'.format(icase, running_score))
	
	# score /= len(dataloader.dataset)
	# print('Average dice {:.4f}.'.format(score))
	precision /= len(dataloader.dataset)
	recall /= len(dataloader.dataset)
	f1 /= len(dataloader.dataset)
	for ic in range(len(channels)):
		ith = np.argmax(f1[ic])
		print('Channel {} has maximum average precision {:.4f}, recall {:.4f}, f1 {:.4f} at threshold {}.'.format(channels[ic], precision[ic, ith], recall[ic, ith], f1[ic, ith], ths[ith]))
	
	print('Testing completed.')


if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	args = get_args()
	os.makedirs(args.result_path, exist_ok=True)

	from data import HistoDataset
	testls_file = 'test_list.txt'
	test_set = HistoDataset(testls_file, phase='test')

	dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

	from model import Unet
	model = Unet(args.in_channels, args.out_channels)
	model.load_state_dict(torch.load(args.model_file))

	from criteria import prf, dice, multichannel_prcurve

	try:
		test_model(model, dataloader, multichannel_prcurve)
	except KeyboardInterrupt:
		sys.exit(0)
