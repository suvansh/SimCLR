import os
import pickle
import numpy as np
from DIP.utils.common_utils import np_to_pil, pil_to_np
from utils import image_hash


class DIPTransformPreComputed:
    def __init__(self, num_transforms, transforms_dir, transforms_iter_file, delimiter='|'):
        """
        :param num_transforms: int representing number of transforms stored per image in dataset
        :param transforms_dir: string representing path to a folder with the following structure:
        if `img` is the np array representation of an image from the dataset,
        let `hash_ = hash(pickle.dumps(img[::4, ::4].tolist()))`, `num` be an integer in [1, num_transforms]
        in file transforms_path/hash_/num.npy, there is a numpy save file containing the <num>th
        pre-computed transformation of `img`
        :param transforms_iter_file: string representing path to file containing DELIMITER-separated CSV with two columns, transform number and DIP iteration number
        :param delimiter: used above
        """
        self.num_transforms = num_transforms
        self.transforms_dir = transforms_dir
        self.transforms_iter_map = {}
        #with open(transforms_iter_file, 'r') as f:
        #    f.readline()  # skip header
        #    while line := f.readline():
        #        t_num, i_num = line.split(delimiter)
        #        t_num, i_num = int(t_num.strip()), int(i_num.strip())
        #        self.transforms_iter_map[t_num] = i_num

    def __call__(self, sample):
        """
        Takes in SAMPLE, a PIL image. Returns pre-computed transform of SAMPLE.
        """
        np_img = pil_to_np(sample)
        hash_ = image_hash(np_img) 
        t_num = np.random.randint(1, self.num_transforms+1)
        i_num = self.transforms_iter_map[t_num]
        transform_path = os.path.join(self.transforms_path, hash_, f'{t_num}_{i_num}.npy')
        np_transformed = np.load(transform_path)
        return np_to_pil(transformed)

