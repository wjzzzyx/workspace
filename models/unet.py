import math
from collections import OrderedDict
import torch
import torch.nn.functional as F

class Unet(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()

		# Encoder stage 1
		self.en1 = torch.nn.Sequential(OrderedDict([
			('conv1_1', torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)),
			('bn1_1', torch.nn.BatchNorm2d(num_features=64)),
			('relu1_1', torch.nn.ReLU(inplace=True)),
			('conv1_2', torch.nn.Conv2d(64, 64, 3, padding=1)),
			('bn1_2', torch.nn.BatchNorm2d(64)),
			('relu1_2', torch.nn.ReLU(inplace=True))
		]))
		self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

		# Encoder stage 2
		self.en2 = torch.nn.Sequential(OrderedDict([
			('conv2_1', torch.nn.Conv2d(64, 128, 3, padding=1)),
			('bn2_1', torch.nn.BatchNorm2d(128)),
			('relu2_1', torch.nn.ReLU(inplace=True)),
			('conv2_2', torch.nn.Conv2d(128, 128, 3, padding=1)),
			('bn2_2', torch.nn.BatchNorm2d(128)),
			('relu2_2', torch.nn.ReLU(inplace=True))
		]))
		self.pool2 = torch.nn.MaxPool2d(2)

		# Encoder stage 3
		self.en3 = torch.nn.Sequential(OrderedDict([
			('conv3_1', torch.nn.Conv2d(128, 256, 3, padding=1)),
			('bn3_1', torch.nn.BatchNorm2d(256)),
			('relu3_1', torch.nn.ReLU(inplace=True)),
			('conv3_2', torch.nn.Conv2d(256, 256, 3, padding=1)),
			('bn3_2', torch.nn.BatchNorm2d(256)),
			('relu3_2', torch.nn.ReLU(inplace=True))
		]))
		self.pool3 = torch.nn.MaxPool2d(2)

		# Encoder stage 4
		self.en4 = torch.nn.Sequential(OrderedDict([
			('conv4_1', torch.nn.Conv2d(256, 512, 3, padding=1)),
			('bn4_1', torch.nn.BatchNorm2d(512)),
			('relu4_1', torch.nn.ReLU(inplace=True)),
			('conv4_2', torch.nn.Conv2d(512, 512, 3, padding=1)),
			('bn4_2', torch.nn.BatchNorm2d(512)),
			('relu4_2', torch.nn.ReLU(inplace=True))
		]))
		self.pool4 = torch.nn.MaxPool2d(2)

		# Encoder stage 5
		self.en5 = torch.nn.Sequential(OrderedDict([
			('conv5_1', torch.nn.Conv2d(512, 1024, 3, padding=1)),
			('bn5_1', torch.nn.BatchNorm2d(1024)),
			('relu5_1', torch.nn.ReLU(inplace=True)),
			('conv5_2', torch.nn.Conv2d(1024, 1024, 3, padding=1)),
			('bn5_2', torch.nn.BatchNorm2d(1024)),
			('relu5_2', torch.nn.ReLU(inplace=True))
		]))

		# Decoder stage 1
		self.up1 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
		self.de1 = torch.nn.Sequential(
			torch.nn.Conv2d(512 * 2, 512, 3, padding=1),
			torch.nn.BatchNorm2d(512),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(512, 512, 3, padding=1),
			torch.nn.BatchNorm2d(512),
			torch.nn.ReLU(inplace=True)
		)

		# Decoder stage 2
		self.up2 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
		self.de2 = torch.nn.Sequential(
			torch.nn.Conv2d(256 * 2, 256, 3, padding=1),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(256, 256, 3, padding=1),
			torch.nn.BatchNorm2d(256),
			torch.nn.ReLU(inplace=True)
		)

		# Decoder stage 3
		self.up3 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
		self.de3 = torch.nn.Sequential(
			torch.nn.Conv2d(128 * 2, 128, 3, padding=1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(128, 128, 3, padding=1),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(inplace=True),
		)

		# Decoder stage 4
		self.up4 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
		self.de4 = torch.nn.Sequential(
			torch.nn.Conv2d(64 * 2, 64, 3, padding=1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(64, 64, 3, padding=1),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(inplace=True)
		)

		self.outlayer = torch.nn.Conv2d(64, out_channels, 1)

	def forward(self, x):
		h1 = self.en1(x)
		p1 = self.pool1(h1)
		h2 = self.en2(p1)
		p2 = self.pool2(h2)
		h3 = self.en3(p2)
		p3 = self.pool3(h3)
		h4 = self.en4(p3)
		p4 = self.pool4(h4)
		h5 = self.en5(p4)
		u1 = self.up1(h5)
		c1 = self.concat(u1, h4)
		h6 = self.de1(c1)
		u2 = self.up2(h6)
		c2 = self.concat(u2, h3)
		h7 = self.de2(c2)
		u3 = self.up3(h7)
		c3 = self.concat(u3, h2)
		h8 = self.de3(c3)
		u4 = self.up4(h8)
		c4 = self.concat(u4, h1)
		h9 = self.de4(c4)
		out = self.outlayer(h9)

		return out
	
	@staticmethod
	def concat(a, b):
		diffh = b.size()[2] - a.size()[2]
		diffw = b.size()[3] - a.size()[3]
		a = F.pad(a, (math.floor(diffw / 2), math.ceil(diffw / 2), math.floor(diffh / 2), math.ceil(diffh / 2)))
		return torch.cat([a, b], dim=1)

