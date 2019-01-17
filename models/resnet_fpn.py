import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class Resnet_fpn(nn.Module):
	
	def __init__(self, backbone, in_channels, max_channels, out_channels):
		super().__init__()

		# copy layers from backbone resnet
		self.conv1 = backbone.conv1
		self.bn1 = backbone.bn1
		self.relu = backbone.relu
		self.maxpool = backbone.maxpool
		self.layer1 = backbone.layer1
		self.layer2 = backbone.layer2
		self.layer3 = backbone.layer3
		self.layer4 = backbone.layer4

		# propagate high layer signals back to low layers
		self.shortcut1 = nn.Conv2d(max_channels, 256 , 1)
		self.shortcut2 = nn.Conv2d(max_channels // 2, 256, 1)
		self.absorb2 = nn.Conv2d(512, 256, 3, padding=1)
		self.shortcut3 = nn.Conv2d(max_channels // 4, 256, 1)
		self.absorb3 = nn.Conv2d(512, 256, 3, padding=1)
		self.shortcut4 = nn.Conv2d(max_channels // 8, 256, 1)
		self.absorb4 = nn.Conv2d(512, 256, 3, padding=1)

		# deconv layers
		for i in range(1, 5):
			setattr(self, 'de' + str(i), torch.nn.Sequential(
				torch.nn.Conv2d(256, 128, 3, padding=1),
				torch.nn.BatchNorm2d(128),
				torch.nn.ReLU(inplace=True),
				torch.nn.Conv2d(128, 128, 3, padding=1),
				torch.nn.BatchNorm2d(128),
				torch.nn.ReLU(inplace=True)
				)
			)
		
		# convs after concat pyramid features
		self.pyramid_merge = nn.Sequential(
			torch.nn.Conv2d(512, 256, 3, padding=1),
			torch.nn.BatchNorm(256),
			torch.nn.ReLU(inplace=True),
			torch.nn.Upsample(scale_factor=2, mode='bilinear'),
			torch.nn.Conv2d(256, 128, 3, padding=1),
			torch.nn.ReLU(inplace=True)
		)

		# final conv layers before output
		self.output = nn.Sequential(
			torch.nn.Conv2d(128, 128, 3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.Upsample(scale_factor=2, mode='bilinear'),
			torch.nn.Conv2d(128, 64, 3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(64, 64, 3, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(64, out_channels, 1)
		)
	
	def foward(self, x):
		c1 = self.relu(self.bn1(self.conv1(x)))
		x = self.maxpool(c1)
		c2 = self.layer1(x)
		c3 = self.layer2(c2)
		c4 = self.layer3(c3)
		c5 = self.layer4(c4)
		# propagate high layer signals back to low layers
		p5 = self.shortcut1(c5)
		p5_upsample = F.upsample(p5, scale_factor=2, mode='bilinear')
		p4 = self.shortcut2(c4)
		p4 = p4 + p5_upsample
		p4 = self.absorb2(p4)
		p4_upsample = F.upsample(p4, scale_factor=2, mode='bilinear')
		p3 = self.shortcut3(c3)
		p3 = p3 + p4_upsample
		p3 = self.absorb3(p3)
		p3_upsample = F.upsample(p3, scale_factor=2, mode='bilinear')
		p2 = self.shortcut4(c2)
		p2 = p2 + p3_upsample
		p2 = self.absorb4(p2)
		# deconv layers
		d1 = self.de1(p5)
		d2 = self.de2(p4)
		d3 = self.de3(p3)
		d4 = self.de4(p2)
		h = torch.cat([
			F.upsample(d1, scale_factor=8, mode='bilinear'),
			F.upsample(d2, scale_factor=4, mode='bilinear'),
			F.upsample(d3, scale_factor=2, mode='bilinear'),
			d4
		], dim=1)
		h = self.pyramid_merge(h)
		h = torch.cat([c1, h], dim=1)
		self.out = self.output(h)

		return self.out


def resnet18_fpn(pretrained=False, in_channels, out_channels):
	base = resnet18(pretrained, in_channels)
	model = ResNet_fpn(base, in_channels, 512, out_channels)
	if pretrained:
		pretrained_dict = base.state_dict()
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	return model


def resnet34_fpn(pretrained=False, in_channels, out_channels):
	base = resnet34(pretrained, in_channels)
	model = ResNet_fpn(base, in_channels, 512, out_channels)
	if pretrained:
		pretrained_dict = base.state_dict()
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	return model


def resnet50_fpn(pretrained=False, in_channels, out_channels):
	base = resnet50(pretrained, in_channels)
	model = ResNet_fpn(base, in_channels, 2048, out_channels)
	if pretrained:
		pretrained_dict = base.state_dict()
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	return model


def resnet101_fpn(pretrained=False, in_channels, out_channels):
	base = resnet101(pretrained, in_channels)
	model = ResNet_fpn(base, in_channels, 2048, out_channels)
	if pretrained:
		pretrained_dict = base.state_dict()
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	return model


def resnet152_fpn(pretrained=False, in_channels, out_channels):
	base = resnet152(pretrained, in_channels)
	model = ResNet_fpn(base, in_channels, 2048, out_channels)
	if pretrained:
		pretrained_dict = base.state_dict()
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
	return model
