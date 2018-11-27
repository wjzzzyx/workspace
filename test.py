import os
import sys
import argparse
from PIL import Image
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
	precision = 0.0
	recall = 0.0
	f1 = 0.0
	for icase, (image, label) in enumerate(dataloader):
		image = image.to(device)
		with torch.no_grad():
			pred = model(image)
			pred = F.softmax(pred, dim=1).squeeze_()
		pos_map = pred[1].cpu().numpy()
		label = label.squeeze_().numpy()
		running_score = criterion(label, pos_map, 0.5)
		score += running_score
		# res = criterion(label, pos_map, 0.5)
		# running_p = res['precision']
		# running_r = res['recall']
		# running_f = res['f1_score']
		# precision += running_p
		# recall += running_r
		# f1 += running_f
		pos_map[pos_map >= 0.5] = 1
		pos_map[pos_map < 0.5] = 0
		pos_map = (pos_map * 255).astype(np.uint8)
		pos_map = Image.fromarray(pos_map)
		pos_map.save(os.path.join(args.result_path, str(icase) + '.png'))
		# print('Testing image {}, dice {:.4f}.'.format(icase, running_score))
		# print('Testing image {}, precision {:.4f}, recall {:.4f}, f1 {:.4f}.'.format(icase, running_p, running_r, running_f))
	score /= len(dataloader.dataset)
	print('Average dice {:.4f}.'.format(score))
	# precision /= len(dataloader.dataset)
	# recall /= len(dataloader.dataset)
	# f1 /= len(dataloader.dataset)
	# print('Average precision {:.4f}, recall {:.4f}, f1 {:.4f}.'.format(precision, recall, f1))
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

	from criteria import prf, dice

	try:
		test_model(model, dataloader, dice)
	except KeyboardInterrupt:
		sys.exit(0)
