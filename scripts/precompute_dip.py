import os
import pickle
import shutil
import multiprocessing as mp
from utils import get_repo_dir, image_hash, chunk
from DIP.utils.common_utils import * 
from DIP.models.skip import skip
from DIP.models import get_net
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from data_transforms import identity_transform

class DIPCompute:
    def __init__(self, dataloader, save_dir, image_shape, num_loops, iters,
                 input_depth=32, input_noise_std=0, optimizer='adam', lr=1e-2,
                 device='cpu'):
        """
        :param dataloader: the dataloader containing images to operate on.
        :param save_dir: str containing path to save subdirectories with npy files to.
        :param image_shape: image shape (CxHxW)
        :param num_loops: number of loops to do over the dataset (for more randomness)
        :param iters: the iterations to save at in each run.
        :param input_depth: depth of random input. Default value taken from paper.
        :param input_noise_std: stdev of noise added to base random input at each iter. They say in the paper that this helps sometimes (seemingly using 0.03), but default is 0 (no noise).
        :param optimizer: supported optimizers are 'adam' and 'LBFGS'
        :param lr: learning rate. Per paper and code, 1e-2 works best.
        :device: 'cuda' to use GPU if available.
        For example, with NUM_ITERS = 100 and STOP_ITERS = {10: 0.5, 50: 0.3, 100: 0.2}, there is a 50% chance that the transform runs for 10 iterations, a 30% chance it runs for 50 iterations, and a 20% chance it runs for the full 100 iterations. 
        """
        self.iters = sorted(iters)
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.num_iters = self.iters[-1] + 1  # do 1 more than the max since 0-indexing
        self.num_loops = num_loops
        self.image_shape = image_shape
        self.input_depth = input_depth
        self.input_noise_std = input_noise_std
        self.device = device
        self.optimizer = optimizer
        self.lr = lr
        self.loss = torch.nn.MSELoss()
        # const strings from provided DIP code
        self.opt_over = 'net'
        self.const_input = 'noise'

        self.iter_idx = 0
        self.loop_count = 0
        
        # initialize network
        self.net = skip(
            input_depth, 3, 
            num_channels_down = [8, 16, 32], 
            num_channels_up   = [8, 16, 32],
            num_channels_skip = [0, 0, 4], 
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True,
            pad='zeros', act_fun='LeakyReLU'
        ).to(self.device)

    def before_each_img(self, image):
        """ Some initial prep before each image. """
        self.iter_idx = 0
        hash_ = image_hash(image)
        self.cur_save_dir = os.path.join(self.save_dir, hash_)
        # TODO remove. temp modification to resume script after stopping
        if os.path.isdir(self.cur_save_dir):
            return False
        else:
            os.makedirs(self.cur_save_dir)
            return True
        

    def save(self, iter_num, np_img):
        transform_num = self.loop_count * len(self.iters) + self.iter_idx
        with open(os.path.join(self.cur_save_dir, f'{transform_num:03}_{iter_num:04}.npy'), 'wb') as f:
            np.save(f, np_img)

    def run(self, image):
        assert image.shape[1:] == self.image_shape, 'Wrong shape. Expected {}, got {}.'.format(self.image_shape, image.shape[1:])
        if not self.before_each_img(image):
            return
        for self.loop_count in range(self.num_loops):
            # run net for num_iters iterations
            net_input = get_noise(self.input_depth, self.const_input, self.image_shape[1:]).to(self.device)
            net_input_saved = net_input.detach().clone()
            noise = net_input.detach().clone()  # dummy tensor of same shape
            p = get_params(self.opt_over, self.net, net_input)
            lambda_closure = lambda iter_num: self.closure(net_input_saved, image, noise, iter_num)
            optimize(self.optimizer, p, lambda_closure, self.lr, self.num_iters, pass_iter=True)
        
    def closure(self, net_input_saved, image, noise, iter_num):
        net_input = net_input_saved
        if self.input_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * self.input_noise_std)
        out = self.net(net_input)
        total_loss = self.loss(out, image)
        total_loss.backward()
        if iter_num == self.iters[self.iter_idx]:
            self.iter_idx += 1
            self.save(iter_num, torch_to_np(out))
            if self.iter_idx == len(self.iters):
                self.iter_idx = 0

    def run_all(self):
        for image, target in tqdm(self.dataloader):
            self.run(image.to(self.device))

data = CIFAR10(root=os.path.join(get_repo_dir(), 'data'), train=True, download=True, transform=identity_transform)
save_dir = os.path.join(get_repo_dir(), 'data/transforms')
os.makedirs(save_dir, exist_ok=True)

#first = True
#hashes = set()
#if first:
#    for im, t in data:
#        hashes.add(image_hash(im))
#    assert len(data) == len(hashes), 'colliding hash'
#    pickle.dump(hashes, open(os.path.join(save_dir, 'hashes.pkl'), 'wb'))
#else:
#    hashes_old = pickle.load(open(os.path.join(save_dir, 'hashes.pkl'), 'rb'))
#    for im, t in data:
#        hashes.add(hash_ := image_hash(im))
#        assert hash_ in hashes_old, 'inconsistent hash'
#import pdb; pdb.set_trace()


CONST_NUM_LOOPS = 5
iters = list(range(20, 201, 10)) + list(range(300, 1001, 50)) + list(range(1100, 1601, 100))
with open(os.path.join(save_dir, 'iter_info.txt'), 'w') as f:
    f.write('Transform Num | DIP Iter Num\n')
    for loop in range(CONST_NUM_LOOPS):    
        for idx, itr in enumerate(iters):
            f.write(f'{idx+1+loop*len(iters):4} | {itr:4}\n')

def proc_func(data_idx, idx):
    """ Apply the DIPCompute transform to DATA and store it. IDX used for multiprocessing. """
    data_subset = Subset(data, data_idx)
    dataloader = DataLoader(data_subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    gpu_num = idx % (torch.cuda.device_count() - 1) + 1
    comp = DIPCompute(dataloader, save_dir, (3, 32, 32), num_loops=CONST_NUM_LOOPS, iters=iters, input_noise_std=0.03, device=f'cuda:{gpu_num}')
    comp.run_all()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    nproc = 32
    data_idxs = list(range(len(data)//2, len(data)))  # TODO do other half
    chks = chunk(data_idxs, nproc)
    with mp.Pool(processes=nproc) as pool:
        for chk in chks:
            results = []
            for chk_idx, chk in enumerate(chks):
                #subset_data = Subset(data, chk)
                results.append(pool.apply_async(proc_func, (chk, chk_idx)))
            for res in results:
                res.get()

