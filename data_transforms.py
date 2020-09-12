import os
from torchvision import transforms
from transforms.dip import DIPTransform
from transforms.dip_precomputed import DIPTransformPreComputed
from utils import get_repo_dir
from DIP.utils.common_utils import pil_to_np, np_to_pil
import cv2

identity_transform = transforms.Compose([
    transforms.ToTensor(),
])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

DIP_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    DIPTransform((3, 32, 32), 2000, {30: 0.1, 50: 0.2, 70: 0.4, 90: 0.2, 100: 0.1},
                 input_noise_std=0.03, plot_every=0, device='cuda'),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

gaussian_blur_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.Lambda(lambda img: pil_to_np(img)),
    transforms.Lambda(lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
    transforms.Lambda(lambda img: np_to_pil(img)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

random_blur_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.Lambda(lambda img: pil_to_np(img)),
    transforms.RandomChoice([
        transforms.Lambda(lambda img: cv2.blur(img, (5, 5))),
        transforms.Lambda(lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
        transforms.Lambda(lambda img: cv2.medianBlur(img, 5)),
    ]),
    transforms.Lambda(lambda img: np_to_pil(img)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transforms_dir = os.path.join(get_repo_dir(), 'data/transforms')
DIP_transform_precomputed = transforms.Compose([
    DIPTransformPreComputed(200, transforms_dir, os.path.join(transforms_dir, 'iter_info.txt')),
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
