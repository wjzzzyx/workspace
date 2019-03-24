import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import GANLoss
from models.resnet_fpn import resnet50_fpn, resnet152_fpn
from models.nlayerdiscriminator import NLayerDiscriminator
from models.nlayercnn import NLayerCNN


class SegmentThenClassifyModel():

    def __init__(self, **kwargs):
        self.segnet_in_channels = kwargs['segnet_in_channels']
        self.segnet_out_channels = kwargs['segnet_out_channels']
        self.clsnet_in_channels = kwargs['clsnet_in_channels']
        self.clsnet_out_channels = kwargs['clsnet_out_channels']
        self.phase = kwargs['phase']
        self.device = kwargs['device']
        self.gpus = [int(x) for x in list(kwargs['gpus'])]
        
        if self.phase == 'train':
            use_sigmoid = not kwargs['use_lsgan']
            self.segnet = resnet50_fpn(self.segnet_in_channels, self.segnet_out_channels, pretrained=False)
            self.netD = NLayerDiscriminator(self.segnet_in_channels + self.segnet_out_channels, 64, use_sigmoid=use_sigmoid, init_type='normal')
            self.clsnet = NLayerCNN(self.clsnet_in_channels, self.clsnet_out_channels, 64)
            if len(self.gpus) > 1:
                self.segnet = nn.DataParallel(self.segnet, device_ids=self.gpus)
                self.netD = nn.DataParallel(self.netD, device_ids=self.gpus)
                self.clsnet = nn.DataParallel(self.clsnet, device_ids=self.gpus)
            self.segnet.to(self.device)
            self.netD.to(self.device)
            self.clsnet.to(self.device)
        else:
            print('Loading model from {}.'.format(kwargs['segnet_model_file']))
            self.segnet = resnet152_fpn(self.segnet_in_channels, self.segnet_out_channels, pretrained=False)
            self.segnet.load_state_dict(torch.load(kwargs['segnet_model_file']))
            self.segnet.to(self.device)
            self.segnet.eval()
            print('Loading model from {}.'.format(kwargs['clsnet_model_file']))
            self.clsnet = NLayerCNN(self.clsnet_in_channels, self.clsnet_out_channels, 64)
            self.clsnet.load_state_dict(torch.load(kwargs['clsnet_model_file']))
            self.clsnet.to(self.device)
            self.clsnet.eval()

        if self.phase == 'train':
            # self.fake_AB_pool = ImagePool(kwargs['poolsize'])
            
            self.GANloss = GANLoss(use_lsgan=kwargs['use_lsgan'])
            self.L1loss = nn.L1Loss()
            self.lambda_L1 = kwargs['lambda_L1']
            self.CEloss = nn.CrossEntropyLoss()

            self.optimizer_seg = torch.optim.Adam(self.segnet.parameters(), lr=kwargs['segnet_lr'], betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=kwargs['segnet_lr'], betas=(0.5, 0.999))
            self.optimizer_cls = torch.optim.Adam(self.clsnet.parameters(), lr=kwargs['clsnet_lr'], betas=(0.5, 0.999))
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - kwargs['niter']) / float(kwargs['niter_decay'] + 1)
                return lr_l
            self.scheduler_seg = torch.optim.lr_scheduler.LambdaLR(self.optimizer_seg, lr_lambda=lambda_rule)
            self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_rule)
            self.scheduler_cls = torch.optim.lr_scheduler.LambdaLR(self.optimizer_cls, lr_lambda=lambda_rule)

    def set_input(self, input):
        self.seg_input = input['DAPI'].to(self.device)
        self.seg_label = input['seg_label'].to(self.device)
        self.cls_input = input['other_channels'].to(self.device)
        self.cls_label = input['cls_label'].to(self.device)
        # self.seg_label_flat = torch.argmax(self.seg_label, dim=1).type(torch.int64)
    
    def optimize_segnet(self):
        seg_output_logits = self.segnet(self.seg_input)
        self.seg_output = F.softmax(seg_output_logits, dim=1)

        # bp on netD
        self.set_autograd(self.netD, True)
        self.optimizer_D.zero_grad()
        fake_AB = torch.cat([self.seg_input, self.seg_output], 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.GANloss(pred_fake, target_is_real=False)
        real_AB = torch.cat([self.seg_input, self.seg_label], 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.GANloss(pred_real, target_is_real=True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()
        self.optimizer_D.step()

        # bp on segnet
        self.set_autograd(self.netD, False)
        self.optimizer_seg.zero_grad()
        fake_AB = torch.cat([self.seg_input, self.seg_output], 1)
        pred_fake = self.netD(fake_AB)
        self.loss_seg_GAN = self.GANloss(pred_fake, target_is_real=True)
        self.loss_seg_L1 = self.L1loss(self.seg_output[:, 1], self.seg_label[:, 1]) * self.lambda_L1
        # self.loss_seg_CE = self.CEloss(seg_output_logits, self.seg_label_flat)
        self.loss_seg = self.loss_seg_GAN + self.loss_seg_L1
        self.loss_seg.backward()
        self.optimizer_seg.step()

    def optimize_clsnet(self):
        seg_output_logits = self.segnet(self.seg_input)
        self.seg_output = F.softmax(seg_output_logits, dim=1)
        cls_output_logits = self.clsnet(torch.cat([self.seg_output, self.cls_input], 1))
        self.cls_output = F.softmax(cls_output_logits, dim=1)

        # bp on clsnet
        self.loss_cls = self.CEloss(self.cls_output, self.cls_label)
        self.loss_cls.backward()
        self.optimizer_cls.step()

    def set_autograd(self, model, requires_grad):
        if not model is None:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def update_seg_lr(self):
        self.scheduler_seg.step()
        self.scheduler_D.step()
        lr = self.optimizer_seg.param_groups[0]['lr']
        print('learning rate = {:.7f}'.format(lr))

    def update_cls_lr(self):
        self.scheduler_cls.step()
        lr = self.optimizer_cls.param_groups[0]['lr']
        print('learning rate = {:.7f}'.format(lr))

    def predict(self):
        seg_output_logits = self.segnet(self.seg_input)
        self.seg_output = F.softmax(seg_output_logits, dim=1)
        cls_output_logits = self.clsnet(torch.cat([self.seg_output, self.cls_input], 1))
        self.cls_output = F.softmax(cls_output_logits, dim=1)
        return self.seg_output, self.cls_output

    def save(self, save_path, suffix):
        # save the original module state dict
        if len(self.gpus) > 1:
            segnet_state_dict = self.segnet.module.state_dict()
            netD_state_dict = self.netD.module.state_dict()
            clsnet_state_dict = self.clsnet.module.state_dict()
        else:
            segnet_state_dict = self.segnet.state_dict()
            netD_state_dict = self.netD.state_dict()
            clsnet_state_dict = self.clsnet.state_dict()
        torch.save(segnet_state_dict, os.path.join(save_path, 'segnet_' + suffix + '.pth'))
        torch.save(netD_state_dict, os.path.join(save_path, 'netD_' + suffix + '.pth'))
        torch.save(clsnet_state_dict, os.path.join(save_path, 'clsnet_' + suffix + '.pth'))

#     def get_current_visuals(self):
#         def to_image(tensor):
#             array = tensor.detach().cpu().numpy()
#             # if array.shape[0] == 1:
#             #     array = np.tile(array, (3, 1, 1))
#             array = np.squeeze(array)
#             if array.ndim == 3:
#                 array = np.transpose(array, (1, 2, 0))
#             img = (array * 255).astype(np.uint8)
#             return img
#         return {
#             'seg_output': to_image(self.seg_output[0]),
#             'seg_label': to_image(self.seg_label[0]),
#             'cls_output': to_image(self.cls_output[0][1]),
#             'cls_label': to_image(self.cls_label[0][1]),
#         }
    
    def get_seg_losses(self):
        return {
            'loss_seg_GAN': self.loss_seg_GAN.item(),
            # 'loss_G_CE': self.loss_G_CE.item(),
            'loss_seg_L1': self.loss_seg_L1.item(), 
            'loss_D_real': self.loss_D_real.item(), 
            'loss_D_fake': self.loss_D_fake.item()
        }

    def get_cls_losses(self):
        return {
            'loss_cls': self.loss_cls.item()
        }
