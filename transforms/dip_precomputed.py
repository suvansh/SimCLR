import os
import pickle
import numpy as np
from DIP.utils.common_utils import np_to_pil, pil_to_np
from utils import image_hash


class DIPTransformPreComputed:
    def __init__(self, num_transforms, transforms_path):
        """
        :param num_transforms: int representing number of transforms stored per image in dataset
        :param transforms_path: string representing path to a folder with the following structure:
        if `img` is the np array representation of an image from the dataset,
        let `hash_ = hash(pickle.dumps(img[::4, ::4].tolist()))`, `num` be an integer in [1, num_transforms]
        in file transforms_path/hash_/num.npy, there is a numpy save file containing the <num>th
        pre-computed transformation of `img`
        """
        self.num_transforms = num_transforms
        self.transforms_path = transforms_path

    def __call__(self, sample):
        """
        Takes in SAMPLE, a PIL image. Returns pre-computed transform of SAMPLE.
        """
        np_img = pil_to_np(sample)
        hash_ = image_hash(np_img) 
        rand_iter = np.random.randint(self.num_transforms)
        transform_path = os.path.join(self.transforms_path, hash_, f'{rand_iter}.npy')
        np_transformed = np.load(transform_path)
        return np_to_pil(transformed)

