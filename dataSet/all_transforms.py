# @Author: Xuan Cao <xuan>
# @Date:   2019-12-22, 12:19:37
# @Last modified by:   xuan
# @Last modified time: 2019-12-22, 1:34:23



import random
from .transforms import *
import albumentations as albu
from albumentations.pytorch import ToTensor

def transform_train_al(size):
    if isinstance(size, int):
        H, W = size, size
    else:
        H, W = size
    transform = albu.Compose([
        #albu.RandomCrop(H, W),
#        albu.OneOf([
#            albu.RandomGamma(gamma_limit=(60, 120), p=0.9),
#            albu.RandomBrightnessContrast(brightness_limit=0.2,
#                                          contrast_limit=0.2, p=0.9),
#            albu.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
#        ]),
#        albu.OneOf([
#            albu.Blur(blur_limit=4, p=1),
#            albu.MotionBlur(blur_limit=4, p=1),
#            albu.MedianBlur(blur_limit=4, p=1)
#        ], p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        #albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
#        albu.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0,
#                              interpolation=cv2.INTER_LINEAR,
#                              border_mode=cv2.BORDER_CONSTANT, p=1),
        albu.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensor()
        ])
    return transform

def transform_test_al(size):
    if isinstance(size, int):
        H, W = size, size
    else:
        H, W = size
    transform = albu.Compose([
        albu.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensor()
        ])
    return transform

def mask2contour(mask, width=3):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3)
