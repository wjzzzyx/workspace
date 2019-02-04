import time
import sys
import os
import argparse
import torch
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    # network parameters
    parser.add_argument('--in-channels', type=int, default=1, help='number of input channels.')
    parser.add_argument('--out-channels', type=int, default=1, help='number of output channels.')
    # training parameters
    parser.add_argument('--gpu', type=str, default='', help='device.')
    parser.add_argument('--load', type=str, default='', help='loading path of trained model.')
    parser.add_argument('--niter', type=int, default=40, help='number of epochs with initial learning rate.')
    parser.add_argument('--niter_decay', type=int, default=40, help='number of epochs with declining learning rate.')
    parser.add_argument('--batch-size', type=int, default=12, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate.')
    parser.add_argument('--val-inteval', type=int, default=2, help='number of epochs between evaluating on the validation set.')
    parser.add_argument('--display-freq', type=int, default=10, help='displaying results every this number of batchs.')
    parser.add_argument('--print-freq', type=int, default=20, help='printing losses every this number of batchs.')
    parser.add_argument('--save-freq', type=int, default=20, help='saving the model every this number of epochs.')
    # storage parameters
    parser.add_argument('--data-path', type=str, default='/mnt/ccvl15/yixiao/kaggle/data', help='data path.')
    parser.add_argument('--model-path', type=str, default='/mnt/ccvl15/yixiao/histopathology/nuclei_contour/models', help='path of pretrained model and snapshots.')
    parser.add_argument('--result-path', type=str, default='/mnt/ccvl15/yixiao/histopathology/nuclei_contour/results', help='path of train logs and test results.')
    
    return parser.parse_args()


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    args = get_args()
    os.makedirs(os.path.join(args.model_path, 'snapshots'), exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    save_path = os.path.join(args.model_path, 'snapshots')

    from data import HistoDataset
    trainlist_fname = '/mnt/ccvl15/yixiao/histopathology/train_list3.txt'
    testlist_fname = '/mnt/ccvl15/yixiao/histopathology/test_list3.txt'
    train_set = HistoDataset(trainlist_fname, phase='train', use_data_augmentation=False, use_normalization=True)

    dataloaders = {}
    dataloaders['train'] = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print('Dataset ready. Number of training images {}'.format(len(train_set)))

    from pix2pix_model import Pix2PixModel
    params = {
        'netG_in_channels': args.in_channels,
        'netG_out_channels': args.out_channels,
        'phase': 'train',
        'device': device,
        'gpu': args.gpu,
        'use_lsgan': False,
        'lr': args.lr,
        'niter': args.niter,
        'niter_decay': args.niter_decay,
        'lambda_L1': 10,
    }
    model = Pix2PixModel(**params)
    print('Pix2PixModel is created.')

    from util.visualizer import Visualizer
    params = {
        'display_id': 1,
        'isTrain': True,
        'no_html': True,
        'display_winsize': 256,
        'name': 'NucleiSeg',
        'display_server': 'http://localhost',
        'display_port': 8097,
        'display_env': 'main',
        'display_ncols': 4,
        'result_dir': args.result_path
    }
    visualizer = Visualizer(**params)

    try:
        print('Begin training...')
    
        for epoch in range(1, args.niter + args.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0
            moving_avg_loss = {'loss_G_GAN': 0.0, 'loss_G_L1': 0.0, 'loss_D_real': 0.0, 'loss_D_fake': 0.0}
            for icase, (images, labels) in enumerate(dataloaders['train']):
                visualizer.reset()
                epoch_iter += args.batch_size
                model.set_input({'A': images, 'B': labels})
                model.optimize()
                
                losses = model.get_current_losses()
                for k, v in losses:
                    moving_avg_loss[k] += v
                
                if epoch_iter % (args.batch_size * args.display_freq) == 0:
                    visualizer.display_current_results(model.get_current_visuals(), epoch)
                if total_steps % (args.batch_size * args.print_freq) == 0:
                    visualizer.print_current_losses(epoch, epoch_iter, moving_avg_loss, time.time() - epoch_start_time)
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataloaders['train'].dataset), moving_avg_loss)
                    for k in moving_avg_loss.keys():
                        moving_avg_loss[k] = 0.0
            
            if epoch % args.save_freq == 0:
                model.save(save_path, str(epoch))
    
            model.update_lr()
            print('End of epoch {} / {}, time taken {}s'.format(epoch, args.niter + args.niter_decay, time.time() - epoch_start_time))
        model.save(save_path, 'final')

    except KeyboardInterrupt:
        sys.exit(0)
