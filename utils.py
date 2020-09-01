import os
import pickle
import torch
from PIL import Image
from DIP.utils.common_utils import torch_to_np, pil_to_np
from torchvision.datasets import CIFAR10


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

def get_repo_dir():
    return os.path.dirname(__file__)

def image_hash(image):
    """ Hashes image after subsampling to save time. Returns as str """
    if isinstance(image, torch.Tensor):
        image = torch_to_np(image)
    elif isinstance(image, Image):
        image = pil_to_np(image)
    subsampled = image[:, ::4, ::4]
    hash_ = hash(pickle.dumps(subsampled.tolist()))
    return str(abs(hash_))

def chunk(data, parts):
    divided = [None] * parts
    n = len(data) // parts
    extras = len(data) - n * parts
    prev = 0
    for i in range(parts):
        nxt = prev + n + int(i < extras)
        divided[i] = data[prev:nxt]
        prev = nxt
    return divided

