import os
import numpy as np
from PIL import Image, ImageEnhance
from skimage import io, color
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from albumentations import (
	CLAHE, IAASharpen, IAAEmboss,
	IAAAdditiveGaussianNoise, GaussNoise,
	ToGray,
	Blur, MotionBlur, MedianBlur,
	RandomContrast, RandomBrightness,
	ChannelShuffle,
	RandomRotate90, Flip, RandomScale,
	ElasticTransform, OpticalDistortion, GridDistortion, IAAPerspective, IAAPiecewiseAffine,
	HueSaturationValue,
	RandomCrop,
	OneOf, Compose
)


class HistoDataset(Dataset):
	'''Histopathology Image Dataset'''

	def __init__(self, samplels_fname, phase):
		# self.data_dir = data_dir
		# self.image_dir = os.path.join(self.data_dir, 'images')
		# self.label_dir = os.path.join(self.data_dir, 'labels')
		with open(samplels_fname) as f:
			self.samplelist = f.read().splitlines()
		self.phase = phase
	
	def __len__(self):
		return len(self.samplelist)
	
	def __getitem__(self, idx):
		if self.phase == 'train':
			# image_fname, label_fname = self.samplelist[idx].split(' ')
			image_fname, membrane_fname, cytoplasm_fname = self.samplelist[idx].split(' ')
			image = io.imread(image_fname)[:, :, :3]
			# label = io.imread(label_fname)
			membrane = io.imread(membrane_fname)
			cytoplasm = io.imread(cytoplasm_fname)
			membrane[membrane > 0] = 1
			cytoplasm[cytoplasm > 0] = 1
			label = membrane + cytoplasm * 2
			image, label = self.train_aug(image, label)
			image = np.transpose(image, (2, 0, 1))
			image = image[np.newaxis, 0]
			# label[label > 0] = 1
			return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)
		elif self.phase == 'val' or self.phase == 'test':
			# image_fname, label_fname = self.samplelist[idx].split(' ')
			image_fname, membrane_fname, cytoplasm_fname = self.samplelist[idx].split(' ')
			image = io.imread(image_fname)[:, :, :3]
			# label = io.imread(label_fname)
			membrane = io.imread(membrane_fname)
			cytoplasm = io.imread(cytoplasm_fname)
			membrane[membrane > 0] = 1
			cytoplasm[cytoplasm > 0] = 1
			label = membrane + cytoplasm * 2
			image, label = self.test_aug(image, label)
			image = np.transpose(image, (2, 0, 1))
			image = image[np.newaxis, 0]    # To grayscale
			# label[label > 0] = 1
			return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

	def transform(self, image, label):
		# # Enhance contrast
		# enh = ImageEnhance.Contrast(image)
		# factor = np.round(255 / np.array(image).max())
		# image = enh.enhance(factor)
		# # Sharpness
		# enh = ImageEnhance.Sharpness(image)
		# factor = 2
		# image = enh.enhance(factor)
		# Random crop
		i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(256, 256))
		image = TF.crop(image, i, j, h, w)
		label = TF.crop(label, i, j, h, w)
		# RGB to Grayscale
		image = TF.to_grayscale(image)
		# To Tensor
		image = TF.to_tensor(image)
		label = TF.to_tensor(label).squeeze_().long()
		# Normalize
		# image = TF.normalize(image, mean=[0.5], std=[0.5])
		
		return image, label
		
	def train_aug(self, image, label):
		aug = Compose([
			OneOf([CLAHE(), IAASharpen(), IAAEmboss()], p=0.5),
			# OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
			# OneOf([MotionBlur(p=0.2), MedianBlur(blur_limit=3, p=0.1), Blur(blur_limit=3, p=0.1)], p=0.2),
			RandomContrast(),
			RandomBrightness(),
			# ChannelShuffle(),
			RandomRotate90(),
			Flip(),
			# RandomScale(scale_limit=(0.0, 0.1)),
			OneOf([ElasticTransform(), OpticalDistortion(), GridDistortion(), IAAPiecewiseAffine()], p=0.5),
			# HueSaturationValue(p=0.3),
		], p=0.9)
		augmented = aug(image=image, mask=label)
		augmented = ToGray(p=1)(image=augmented['image'], mask=augmented['mask'])
		augmented = RandomCrop(256, 256)(image=augmented['image'], mask=augmented['mask'])
		image, label = augmented['image'], augmented['mask']

		return image, label
	
	def test_aug(self, image, label):
		aug = ToGray(p=1)
		augmented = aug(image=image)
		return augmented['image'], label
