import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import GANLoss
from models.resnet_fpn import resnet152_fpn
from models.nlayerdiscriminator import NLayerDiscriminator


class Pix2PixModel():

    def __init__(self, **kwargs):
        self.netG_in_channels = kwargs['netG_in_channels']
        self.netG_out_channels = kwargs['netG_out_channels']
        self.phase = kwargs['phase']
        self.device = kwargs['device']
        self.gpus = [int(x) for x in list(kwargs['gpu'])]
        
        if self.phase == 'train':
            use_sigmoid = not kwargs['use_lsgan']
            self.netG = resnet152_fpn(self.netG_in_channels, self.netG_out_channels, pretrained=False)
            self.netD = NLayerDiscriminator(self.netG_in_channels + self.netG_out_channels, 64, use_sigmoid=use_sigmoid, init_type='normal')
            if len(kwargs['gpu']) > 1:
                self.netG = nn.DataParallel(self.netG, device_ids=self.gpus)
                self.netD = nn.DataParallel(self.netD, device_ids=self.gpus)
            self.netG.to(self.device)
            self.netD.to(self.device)
        else:
            self.netG = resnet152_fpn(self.netG_in_channels, self.netG_out_channels, pretrained=False)
            print('Loading model from {}.'.format(kwargs['model_file']))
            # self.netG = nn.DataParallel(self.netG)
            self.netG.load_state_dict(torch.load(kwargs['model_file']))
            # self.netG = self.netG.module
            self.netG.to(self.device)
            self.netG.eval()

        if self.phase == 'train':
            # self.fake_AB_pool = ImagePool(kwargs['poolsize'])
            
            self.GANloss = GANLoss(self.device, use_lsgan=kwargs['use_lsgan'])
            self.L1loss = nn.L1Loss()
            self.lambda_L1 = kwargs['lambda_L1']

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=kwargs['lr'], betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=kwargs['lr'], betas=(0.5, 0.999))
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - kwargs['niter']) / float(kwargs['niter_decay'] + 1)
                return lr_l
            self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_rule)
            self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule)

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
    
    def optimize(self):
        self.fake_B_logits = self.netG(self.real_A)
        self.fake_B = F.tanh(self.fake_B_logits)

        # bp on netD
        self.set_autograd(self.netD, True)
        self.optimizer_D.zero_grad()
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.GANloss(pred_fake, target_is_real=False)
        real_AB = torch.cat([self.real_A, self.real_B], 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.GANloss(pred_real, target_is_real=True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

        # bp on netG
        self.set_autograd(self.netD, False)
        self.optimizer_G.zero_grad()
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.GANloss(pred_fake, target_is_real=True)
        self.loss_G_L1 = self.L1loss(self.fake_B, self.real_B) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_autograd(self, model, requires_grad):
        if not model is None:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        print('learning rate = {:.7f}'.format(lr))

    def predict(self):
        logits = self.netG(self.real_A)
        self.fake_B = F.tanh(logits)
        return self.fake_B

    def save(self, save_path, suffix):
        # save the original module state dict
        if len(self.gpus) > 1:
            netG_state_dict = self.netG.module.state_dict()
            netD_state_dict = self.netD.module.state_dict()
        else:
            netG_state_dict = self.netG.state_dict()
            netD_state_dict = self.netD.state_dict()
        torch.save(netG_state_dict, os.path.join(save_path, 'netG_' + suffix + '.pth'))
        torch.save(netD_state_dict, os.path.join(save_path, 'netD_' + suffix + '.pth'))

    def get_current_visuals(self):
        def to_image(tensor):
            array = tensor.detach().cpu().numpy()
            # if array.shape[0] == 1:
            #     array = np.tile(array, (3, 1, 1))
            array = np.squeeze(array)
            if array.ndim == 3:
                array = np.transpose(array, (1, 2, 0))
            img = ((array + 1) / 2 * 255).astype(np.uint8)
            return img
        return {'real_A': to_image(self.real_A[0]), 'real_B': to_image(self.real_B[0]), 'fake_B': to_image(self.fake_B[0])}
    
    def get_current_losses(self):
        return {'G_GAN': self.loss_G_GAN.item(), 'G_L1': self.loss_G_L1.item(), 'D_real': self.loss_D_real.item(), 'D_fake': self.loss_D_fake.item()}
