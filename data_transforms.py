import os
from torchvision import transforms
from transforms.dip import DIPTransform
from transforms.dip_precomputed import DIPTransformPreComputed
from utils import get_repo_dir

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

train_transform_DIP = transforms.Compose([
    transforms.RandomResizedCrop(32),
    DIPTransform((3, 32, 32), 2000, {30: 0.1, 50: 0.2, 70: 0.4, 90: 0.2, 100: 0.1},
                 input_noise_std=0.03, plot_every=0, device='cuda'),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transforms_dir = os.path.join(get_repo_dir(), 'data/transforms')

#train_transform_DIP_precomputed = transforms.Compose([
#    transforms.RandomResizedCrop(32),
#    DIPTransformPreComputed(200, transforms_dir, os.path.join(transforms_dir, 'iter_info.txt')),
#    transforms.ToTensor(),
#    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
