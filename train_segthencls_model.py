import os
import sys
import time
import argparse
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from util.util import tensor2image


def get_args():
    parser = argparse.ArgumentParser()
    # network parameters
    parser.add_argument('--seg-in-channels', type=int, default=1, help='number of segnet input channels.')
    parser.add_argument('--seg-out-channels', type=int, default=3, help='number of segnet output channels.')
    parser.add_argument('--cls-in-channels', type=int, default=6, help='number of clsnet input channels.')
    parser.add_argument('--cls-out-channels', type=int, default=6, help='number of clsnet output channels.')
    # data parameters
    parser.add_argument('--trainlist-fname', type=str, default='/mnt/ccvl15/yixiao/histopathology/segthencls/train_list.txt', help='trainlist filename.')
    # training parameters
    parser.add_argument('--gpus', type=str, default='', help='device.')
    parser.add_argument('--load', type=str, default='', help='loading path of trained model.')
    parser.add_argument('--niter', type=int, default=10, help='number of epochs with initial learning rate.')
    parser.add_argument('--niter_decay', type=int, default=10, help='number of epochs with declining learning rate.')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size.')
    parser.add_argument('--segnet-lr', type=float, default=0.0001, help='initial learning rate of segnet.')
    parser.add_argument('--clsnet-lr', type=float, default=1e-6, help='initial learning rate of clsnet.')
    parser.add_argument('--val-inteval', type=int, default=2, help='number of epochs between evaluating on the validation set.')
    parser.add_argument('--display-freq', type=int, default=20, help='displaying results every this number of batchs.')
    parser.add_argument('--print-freq', type=int, default=20, help='printing losses every this number of batchs.')
    parser.add_argument('--save-freq', type=int, default=20, help='saving the model every this number of epochs.')
    # storage parameters
    parser.add_argument('--model-path', type=str, default='/mnt/ccvl15/yixiao/histopathology/segthencls/models', help='path of pretrained model and snapshots.')
    parser.add_argument('--result-path', type=str, default='/mnt/ccvl15/yixiao/histopathology/segthencls/results', help='path of train logs and test results.')

    return parser.parse_args()

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = get_args()
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    from data import SegThenClsDataset
    train_set = SegThenClsDataset(args.trainlist_fname, phase='train', use_augmentation=False, use_normalization=False)
    dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print('Dataset ready. Number of training images {}.'.format(len(train_set)))

    from segthencls_model import SegmentThenClassifyModel
    params = {
        'segnet_in_channels': 1,
        'segnet_out_channels': 3,
        'clsnet_in_channels': 6,
        'clsnet_out_channels': 6,
        'phase': 'train',
        'device': device,
        'gpus': args.gpus,
        'use_lsgan': False,
        'lambda_L1': 20,
        'niter': args.niter,
        'niter_decay': args.niter_decay,
        'segnet_lr': args.segnet_lr,
        'clsnet_lr': args.clsnet_lr,
    }
    model = SegmentThenClassifyModel(**params)
    print('SegThenCls Model ready.')

    from util.visualizer import Visualizer
    params = {
        'server': 'http://localhost',
        'port': 8097,
        'env': 'main',
        'name': 'SegThenCls',
        'winsize': 256,
        'ncols': 4,
        'result_dir': args.result_path
    }
    visualizer = Visualizer(**params)

    try:
        print('Begin training...')

        for epoch in range(1, args.niter + args.niter_decay + 1):
            epoch_iter = 0
            epoch_start_time = time.time()
            moving_avg_loss = defaultdict(float)
            
            for icase, (DAPI_img, seg_label, other_img, cls_label) in enumerate(dataloader):
                epoch_iter += 1
                model.set_input({'DAPI': DAPI_img, 'seg_label': seg_label, 'other_channels': other_img, 'cls_label': cls_label})
                model.optimize_segnet()

                losses = model.get_seg_losses()
                for k, v in losses.items():
                    moving_avg_loss[k] += v

                if epoch_iter % args.display_freq == 0:
                    visuals = {
                        'DAPI': tensor2image(DAPI_img[0], one_hot=False),
                        'segment_label': tensor2image(seg_label[0], one_hot=True),
                        'segment_pred': tensor2image(model.seg_output[0], one_hot=True)
                    }
                    visualizer.display_visuals(visuals, win_id=1)
                if epoch_iter % args.print_freq == 0:
                    visualizer.print_current_losses(epoch, epoch_iter, moving_avg_loss, time.time() - epoch_start_time)
                    visualizer.plot_current_losses(epoch + float(epoch_iter * args.batch_size) / len(train_set), moving_avg_loss, win_id=3)
                    for k in moving_avg_loss.keys():
                        moving_avg_loss[k] = 0.0
                
            if epoch % args.save_freq == 0:
                model.save(save_path, str(epoch))

            model.update_seg_lr()
            print('Training Segnet. End of epoch {} / {}, time taken {}s.'.format(epoch, args.niter + args.niter_decay, time.time() - epoch_start_time))
        
        model.save(save_path, 'single')

        for epoch in range(1, args.niter + args.niter_decay + 1):
            epoch_iter = 0
            epoch_start_time = time.time()
            moving_avg_loss = defaultdict(float)

            for icase, (DAPI_img, seg_label, other_img, cls_label) in enumerate(dataloader):
                epoch_iter += 1
                model.set_input({'DAPI': DAPI_img, 'seg_label': seg_label, 'other_channels': other_img, 'cls_label': cls_label})
                model.optimize_clsnet()

                losses = model.get_cls_losses()
                for k, v in losses.items():
                    moving_avg_loss[k] += v

                if epoch_iter % args.display_freq == 0:
                    visuals = {
                        'DAPI': tensor2image(DAPI_img[0], one_hot=False),
                        'class_label': tensor2image(cls_label[0], one_hot=True),
                        'class_pred': tensor2image(model.cls_output[0], one_hot=True)
                    }
                    visualizer.display_visuals(visuals, win_id=4)
                if epoch_iter % args.print_freq == 0:
                    visualizer.print_current_losses(epoch, epoch_iter, moving_avg_loss, time.time() - epoch_start_time)
                    visualizer.plot_current_losses(epoch + float(epoch_iter * args.batch_size) / len(train_set), moving_avg_loss, win_id=6)
                    for k in moving_avg_loss.keys():
                        moving_avg_loss[k] = 0.0
                
            if epoch % args.save_freq == 0:
                model.save(save_path, str(epoch))

            model.update_cls_lr()
            print('Training Clsnet. End of epoch {} / {}, time taken {}s.'.format(epoch, args.niter + args.niter_decay, time.time() - epoch_start_time))
        
        model.save(save_path, 'joint')

    except KeyboardInterrupt:
        sys.exit(0)
