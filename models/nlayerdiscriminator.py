import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
from .spectral_norm import spectral_norm


class NLayerDiscriminator(nn.Module):
    
    def __init__(self, in_channels, num_filters, num_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, init_type='normal'):
        super().__init__()
        if type(norm_layer) == partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
    
        sequence = [
            spectral_norm(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, 4, 2, 1, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, 4, 1, 1, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [spectral_norm(nn.Conv2d(num_filters * nf_mult, 1, 4, 1, 1))]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
    
        self.model = nn.Sequential(*sequence)
        
        if init_type == 'normal':
            init_func = partial(init.normal_, mean=0.0, std=0.02)
        elif init_type == 'xavier':
            init_func = partial(init.xavier_normal, gain=0.02)
        elif init_type == 'kaiming':
            init_func = partial(init.kaiming_normal_, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init_func = partial(init.orthogonal_, gain=0.02)
        else:
            raise NotImplementedError
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_func(m.weight.data)
                if not m.bias is None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, std)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.model(x)
