from DIP.utils.common_utils import *
from DIP.models.skip import skip
from DIP.models import get_net
from PIL import Image
import torch
from torch.distributions.categorical import Categorical


class DIPTransform:
    def __init__(self, image_shape, num_iters, stop_iters,
                 input_depth=32, input_noise_std=0, optimizer='adam', lr=1e-2,
                 plot_every=0, device='cpu'):
        """
        :param image_shape: image shape (CxHxW)
        :param num_iters: number of iterations to overfit
        :param stop_iters: dict int->float specifying categorical distribution over percentages in (0, 100] representing the probability that the transform runs for KEY/100 * NUM_ITERS iters.
        :param input_depth: depth of random input. Default value taken from paper.
        :param input_noise_std: stdev of noise added to base random input at each iter. They say in the paper that this helps sometimes (seemingly using 0.03), but default is 0 (no noise).
        :param optimizer: supported optimizers are 'adam' and 'LBFGS'
        :param lr: learning rate. Per paper and code, 1e-2 works best.
        :param plot_every: how often to save images. Doesn't save images if set to 0 (default).
        :device: 'cuda' to use GPU if available.
        For example, with NUM_ITERS = 100 and STOP_ITERS = {10: 0.5, 50: 0.3, 100: 0.2}, there is a 50% chance that the transform runs for 10 iterations, a 30% chance it runs for 50 iterations, and a 20% chance it runs for the full 100 iterations. 
        """
        self.iters, probs = zip(*stop_iters.items())
        self.probs = Categorical(torch.tensor(probs))
        self.num_iters = num_iters
        self.image_shape = image_shape
        self.input_depth = input_depth
        self.input_noise_std = input_noise_std
        self.plot_every = plot_every
        self.device = device
        self.optimizer = optimizer
        self.lr = lr
        self.loss = torch.nn.MSELoss()
        # const strings from provided DIP code
        self.opt_over = 'net'
        self.const_input = 'noise'
        
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

    def sample_iters(self):
        return self.iters[self.probs.sample()] * self.num_iters // 100

    def run(self, image, num_iters):
        assert image.shape[1:] == self.image_shape, 'Wrong shape. Expected {}, got {}.'.format(self.image_shape, image.shape[1:])
        # run net for num_iters iterations
        net_input = get_noise(self.input_depth, self.const_input, self.image_shape[1:]).to(self.device)
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        p = get_params(self.opt_over, self.net, net_input)
        #def local_closure(iter_num):
        #    self.closure(net_input_saved, image, noise, iter_num)
        lambda_closure = lambda iter_num: self.closure(net_input_saved, image, noise, iter_num)
        optimize(self.optimizer, p, lambda_closure, self.lr, num_iters, pass_iter=True)
        transformed = self.net(net_input)
        return transformed
        
    def closure(self, net_input_saved, image, noise, iter_num):
        net_input = net_input_saved
        if self.input_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * self.input_noise_std)
        out = self.net(net_input)
        total_loss = self.loss(out, image)
        total_loss.backward()
        #print(total_loss)
        if self.plot_every > 0 and iter_num % self.plot_every == 0 and total_loss < 0.01:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], factor=4, nrow=1, show=False, save_path=f'results_dip/imgs/{iter_num}.png')
        # maybe log loss here?

    def __call__(self, sample):
        """
        Takes in, transforms, and returns PIL image given by SAMPLE. Transformation is a random number of iterations of DIP.
        Distribution of number of iterations is specified when the transform is initialized.
        """
        torch_img = np_to_torch(tmp := pil_to_np(sample)).to(self.device)
        plot_image_grid([np.clip(tmp, 0, 1)], factor=4, nrow=1, show=False, save_path='results_dip/imgs/true.png')  # TODO remove
        num_iters = self.sample_iters()
        transformed = self.run(torch_img, num_iters)
        return np_to_pil(torch_to_np(transformed))

