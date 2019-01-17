import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.ort/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, channels, stride=1, downsample=None):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if not self.downsample is None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_channels, channels, stride=1, downsample=None):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, bias=False)
		self.bn1 = nn.BatchNorm2d(num_features=channels)
		self.conv2 = nn.Conv2d(channels, channels, 3, stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(channels * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
	
	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)
		if not self.downsample is None:
			identity = self.downsample(x)
		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, in_channels, stage_layernum, num_classes=1000):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_stage(block, 64, 64, stage_layernum[0], False)
		self.layer2 = self._make_stage(block, 64 * 4, 128, stage_layernum[1], True)
		self.layer3 = self._make_stage(block, 128 * 4, 256, stage_layernum[2], True)
		self.layer4 = self._make_stage(block, 256 * 4, 512, stage_layernum[3], True)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		# Initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
	
	def _make_stage(self, block, in_channels, channels, layernum, downsample):
		projection = None
		if downsample:
			projection = nn.Sequential(
				nn.Conv2d(in_channels, channels * block.expansion, 1, stride=2, bias=False),
				nn.BatchNorm2d(channels * block.expansion)
			)
		layers = [block(in_channels, channels, 2, projection)]
		for _ in range(1, layernum):
			layers.append(block(channels * block.expansion, channels))

		return nn.Sequential(*layers)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def resnet18(pretrained=False, in_channels, **kwargs):
	model = ResNet(BasicBlock, in_channels, [2, 2, 2, 2], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
	return model


def resnet34(pretrained=False, in_channels, **kwargs):
	model = ResNet(BasicBlock, in_channels, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
	return model


def resnet50(pretrained=False, in_channels, **kwargs):
	model = ResNet(Bottleneck, in_channels, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def resnet101(pretrained=False, in_channels, **kwargs):
	model = ResNet(Bottleneck, in_channels, [3, 4, 23, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model

def resnet152(pretrained=False, in_channels, **kwargs):
	model = ResNet(Bottleneck, in_channels, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
	return model
