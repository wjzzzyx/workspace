import os
import numpy as np
from PIL import Image, ImageEnhance
from skimage import io, color
import cv2
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

    def __init__(self, datalist_fname, phase, use_data_augmentation=True, use_normalization=True):
        # self.data_dir = data_dir
        # self.image_dir = os.path.join(self.data_dir, 'images')
        # self.label_dir = os.path.join(self.data_dir, 'labels')
        with open(datalist_fname) as f:
            self.datalist = f.read().splitlines()
        self.phase = phase
        self.use_data_aug = use_data_augmentation
        self.use_normalization = use_normalization
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        image_fname, label_fname = self.datalist[idx].split(' ')
        if image_fname[-3:] in ['png']:
            image = cv2.imread(image_fname, cv2.IMREAD_UNCHANGED)
        elif image_fname[-3:] == 'npy':
            image = np.load(image_fname)
        else:
            print('Unsupported data format')
            raise NotImplementedError
        if label_fname[-3:] in ['png']:
            label = cv2.imread(label_fname, cv2.IMREAD_UNCHANGED)
        elif label_fname[-3:] == 'npy':
            label = np.load(label_fname)
        else:
            print('Unsupported data format')
            raise NotImplementedError
        if image.ndim == 2:
            image = np.expand_dims(image, 2)
        if label.ndim == 2:
            label = np.expand_dims(label, 2)
        if self.use_data_aug:
            if self.phase == 'train':
                image, label = self.train_aug(image, label)
            else:
                image, label = self.test_aug(image, label)
        # image = transforms.ToTensor()(image)
        # label = transforms.ToTensor()(label)
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image, dtype=torch.float32)
        label = np.transpose(label, (2, 0, 1))
        label = torch.tensor(label, dtype=torch.float32)
        if self.use_normalization:
            # image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
            image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
        # label = (label - 0.5) * 2
        return image, label
    
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


class SegThenClsDataset(Dataset):
    'SegmentThenClassify Dataset'
    def __init__(self, datalist_fname, phase, use_augmentation=False, use_normalization=False):
        with open(datalist_fname) as f:
            self.datalist = f.read().splitlines()
        self.phase = phase
        self.use_data_aug = use_augmentation
        self.use_normalization = use_normalization

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        DAPI_image_fname, other_image_fname, seg_label_fname, cls_label_fname = self.datalist[idx].split(' ')
        DAPI_img = np.load(DAPI_image_fname)
        DAPI_img = np.expand_dims(DAPI_img, 0)
        other_img = np.load(other_image_fname)
        seg_label = np.load(seg_label_fname)
        cls_label = np.load(cls_label_fname)
        DAPI_img = torch.tensor(DAPI_img, dtype=torch.float32)
        other_img = torch.tensor(other_img, dtype=torch.float32)
        seg_label = torch.tensor(seg_label, dtype=torch.float32)
        cls_label = torch.tensor(cls_label, dtype=torch.int64)
        return DAPI_img, seg_label, other_img, cls_label

