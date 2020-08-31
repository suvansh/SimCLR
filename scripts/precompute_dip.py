import os
from utils import get_repo_dir, image_hash
from DIP.utils.common_utils import * 
from DIP.models.skip import skip
from DIP.models import get_net
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_transforms import train_transform

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
        self.num_iters = self.iters[-1]
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
        os.makedirs(self.cur_save_dir, exist_ok=True)

    def save(self, iter_num, np_img):
        # save the optimization iteration this was produced at
        with open(os.path.join(self.cur_save_dir, 'iter.txt'), 'w') as f:
            f.write(f'{iter_num}')
        with open(os.path.join(self.cur_save_dir, f'{self.iter_idx+1}.npy'), 'wb') as f:
            np.save(f, np_img)

    def run(self, image):
        assert image.shape[1:] == self.image_shape, 'Wrong shape. Expected {}, got {}.'.format(self.image_shape, image.shape[1:])
        self.before_each_img(image)
        # run net for num_iters iterations
        net_input = get_noise(self.input_depth, self.const_input, self.image_shape[1:]).to(self.device)
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()  # dummy tensor of same shape
        p = get_params(self.opt_over, self.net, net_input)
        #def local_closure(iter_num):
        #    self.closure(net_input_saved, image, noise, iter_num)
        lambda_closure = lambda iter_num: self.closure(net_input_saved, image, noise, iter_num)
        optimize(self.optimizer, p, lambda_closure, self.lr, self.num_iters, pass_iter=True)
        transformed = self.net(net_input)
        return transformed
        
    def closure(self, net_input_saved, image, noise, iter_num):
        net_input = net_input_saved
        if self.input_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * self.input_noise_std)
        out = self.net(net_input)
        total_loss = self.loss(out, image)
        total_loss.backward()
        if iter_num == self.iters[self.iter_idx]:
            self.save(iter_num, torch_to_np(out))
            self.iter_idx += 1
        # maybe log loss here?

    def run_all(self):
        import pdb; pdb.set_trace()
        for image, target in tqdm(self.dataloader):
            self.run(image.to(self.device))

#import pdb; pdb.set_trace()
#trainset = CIFAR10(
#            root=os.path.join(get_repo_dir(), 'data'), train=True, download=True, transform=train_transform)
#trainloader = DataLoader(
#            trainset, batch_size=64, shuffle=True, num_workers=2)
#for batch in tqdm(trainloader):
#    pass
#
data = CIFAR10(root=os.path.join(get_repo_dir(), 'data'), train=True, download=True, transform=train_transform)
dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
save_dir = os.path.join(get_repo_dir(), 'data/transforms')
iters = list(range(20, 201, 10)) + list(range(300, 1001, 50)) + list(range(1100, 1601, 100))

comp = DIPCompute(dataloader, save_dir, (3, 32, 32), num_loops=5, iters=iters, input_noise_std=0.03, device='cuda')
comp.run_all()

