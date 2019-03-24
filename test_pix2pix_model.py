import os
import sys
import argparse
import time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument('--in-channels', type=int, default=7, help='number of input channels.')
    parser.add_argument('--out-channels', type=int, default=7, help='number of output channels.')
    # training parameters
    parser.add_argument('--gpu', type=str, default='', help='gpus used in training.')
    # storage parameters
    # parser.add_argument('--data-path', type=str, default='/mnt/ccvl15/yixiao/kaggle/data', help='data path.')
    parser.add_argument('--model-file', type=str, default='/mnt/ccvl15/yixiao/histopathology/nuclei_8c/models/snapshots/netG_7c_final.pth', help='trained model or snapshots.')
    parser.add_argument('--result-path', type=str, default='/mnt/ccvl15/yixiao/histopathology/nuclei_8c/7c_results', help='path of predictions and analysis files.')
    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = get_args()
    os.makedirs(os.path.join(args.result_path, 'preds'), exist_ok=True)

    from data import HistoDataset
    testlist_file = '/mnt/ccvl15/yixiao/histopathology/test_list_8c.txt'
    with open(testlist_file) as f:
        testlist = f.read().splitlines()
    test_set = HistoDataset(testlist_file, phase='test', use_data_augmentation=False, use_normalization=False)
    dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
    print('Dataset ready. Number of testing images {}.'.format(len(test_set)))

    from pix2pix_model import Pix2PixModel
    params = {
        'netG_in_channels': args.in_channels,
        'netG_out_channels': args.out_channels,
        'phase': 'test',
        'device': device,
        'gpu': args.gpu,
        'model_file': args.model_file
    }
    model = Pix2PixModel(**params)
    print('Model loaded.')
    
    from criteria import dice

    # from util.visualizer import Visualizer
    # params = {
    #     'display_id': 1,
    #     'isTrain': True,
    #     'no_html': True,
    #     'display_winsize': 256,
    #     'name': 'NucleiSeg',
    #     'display_server': 'http://localhost',
    #     'display_port': 8097,
    #     'display_env': 'main',
    #     'display_ncols': 6,
    #     'result_dir': args.result_path
    # }
    # visualizer = Visualizer(**params)

    try:
        print('Begin testing...')

        score = 0.0
        for icase, (image, label) in enumerate(dataloader):
            image = image[:, :-1, :, :]
            model.set_input({'A': image, 'B': label})
            pred = model.predict().detach().cpu().numpy()
            pred = pred[0]
            # running_score = dice(label, pred, th=128)
            # score += running_score
            # print('Testing image {}, dice {:.4f}.'.format(icase, running_score))
            name = os.path.basename(testlist[icase].split(' ')[0])[:-4] + '.npy'
            # cv2.imwrite(os.path.join(args.result_path, 'preds', name), pred)
            np.save(os.path.join(args.result_path, 'preds', name), pred)
        # score /= len(dataloader.dataset)
        # print('Average dice {:.4f}.'.format(score))

    except KeyboardInterrupt:
        sys.exit(0)
