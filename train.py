import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F


def get_args():
	parser = argparse.ArgumentParser()
	# network parameters
	parser.add_argument('--in-channels', type=int, default=1, help='number of input channels.')
	parser.add_argument('--out-channels', type=int, default=2, help='number of output channels.')
	# training parameters
	parser.add_argument('--load-caffe-unet', action='store_true', help='boolean of loading weights from original caffe unet or not.')
	parser.add_argument('--load', type=str, default='', help='loading path of trained model.')
	parser.add_argument('--epochs', type=int, default=5, help='number of epochs.')
	parser.add_argument('--batch-size', type=int, default=1, help='batch size.')
	parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate.')
	parser.add_argument('--val-inteval', type=int, default=2, help='number of epochs between evaluating on the validation set.')
	parser.add_argument('--save-inteval', type=int, default=1000, help='number of epochs between saving the model.')
	# storage parameters
	parser.add_argument('--data-path', type=str, default='/mnt/ccvl15/yixiao/kaggle/data', help='data path.')
	parser.add_argument('--model-path', type=str, default='/mnt/ccvl15/yixiao/kaggle/models', help='path of pretrained model and snapshots.')
	
	return parser.parse_args()


def train_model(model, dataloader, loss_func, criterion, optimizer, num_epochs):
	val_history = []
	model.to(device)
	print('Begin training...')

	for epoch in range(1, num_epochs):
		print('--------------------------------')
		print('Epoch {} / {}'.format(epoch, num_epochs - 1))
		
		model.train()
		running_loss = 0.0
		last_epoch_loss = 1.0
		for icase, (images, labels) in enumerate(dataloader['train']):
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				preds = model(images)
				loss = loss_func(preds, labels)
				loss.backward()
				optimizer.step()
			running_loss += loss.item() * images.size(0)
		epoch_loss = running_loss / len(dataloader['train'].dataset)
		print('Epoch averaged loss: {:.4f}'.format(epoch_loss))

		if abs(epoch_loss - last_epoch_loss) < 0.001:
			break
		last_epoch_loss = epoch_loss

		# validation
		if epoch % args.val_inteval == 0:
			model.eval()
			running_score = 0.0
			for image, label in dataloader['val']:
				image = image.to(device)
				with torch.set_grad_enabled(False):
					pred = model(image)
					pred = F.softmax(pred, dim=1)
				pred = pred.cpu().numpy()
				pred = np.sum(pred[0, 1:], axis=0)
				label = np.squeeze(label.numpy())
				score = criterion(label, pred, th=0.5)
				running_score += score
			epoch_score = running_score / len(dataloader['val'].dataset)
			val_history.append(epoch_score)
			print('Epoch validation f1 score: {:.4f}'.format(epoch_score))
		
		if epoch % args.save_inteval == 0:
			save_path = os.path.join(args.model_path, 'snapshots', 'Unet_e{}.pt'.format(epoch))
			torch.save(model.state_dict(), save_path)
		print()
	
	print('Training completed.')
	torch.save(model.state_dict(), os.path.join(args.model_path, 'snapshots', 'Unet_final.pt'))
	return model, val_history


def load_conv_weights(model, target_w):
	state_dict = model.state_dict()
	for k in state_dict.keys():
		if 'conv' in k:
			layer_name = k.split('.')[1]
			if 'weight' in k:
				state_dict[k] = target_w[layer_name + '.weight']
			elif 'bias' in k:
				state_dict[k] = target_w[layer_name + '.bias']
	model.load_state_dict(state_dict)
	return model


if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(device)

	args = get_args()
	os.makedirs(os.path.join(args.model_path, 'snapshots'), exist_ok=True)

	from data import HistoDataset
	trainls_file = 'train_list.txt'
	valls_file = 'test_list.txt'
	train_set = HistoDataset(trainls_file, phase='train')
	val_set = HistoDataset(valls_file, phase='val')

	dataloaders = {}
	dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)

	from model import Unet
	model = Unet(args.in_channels, args.out_channels)
	
	if args.load_caffe_unet:
		caffe_unet_path = '/mnt/ccvl15/yixiao/kaggle/models/pretrained/unet.pt'
		unet_caffemodel_weights = torch.load(caffe_unet_path)
		model = load_conv_weights(model, unet_caffemodel_weights)
	elif args.load != '':
		model.load_state_dict(torch.load(args.load))
	
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

	from losses import cross_entropy_with_soft_dice_2
	# loss_func = torch.nn.CrossEntropyLoss()

	from criteria import dice

	try:
		model, val_history = train_model(model, dataloaders, cross_entropy_with_soft_dice_2, dice, optimizer, args.epochs)
	except KeyboardInterrupt:
		sys.exit(0)
	
	
