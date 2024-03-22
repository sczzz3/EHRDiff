
import math
import numpy as np
import torch
import torch.nn as nn


class NaiveDenoiser(nn.Module):
    def __init__(self,
                model,
                ):

        super().__init__()
        self.model = model

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        return self.model(x, sigma.reshape(-1), y)


class EDMDenoiser(nn.Module):
    def __init__(self,
                model,
                sigma_min,
                sigma_max,
                sigma_data=math.sqrt(1. / 3)
                ):

        super().__init__()

        self.sigma_data = sigma_data
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        c_skip = self.sigma_data ** 2. / \
            (sigma ** 2. + self.sigma_data ** 2.)
        c_out = sigma * self.sigma_data / \
            torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_in = 1. / torch.sqrt(self.sigma_data ** 2. + sigma ** 2.)
        c_noise = .25 * torch.log(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)

        x_denoised = c_skip * x + c_out * out
        return x_denoised


class VDenoiser(nn.Module):
    def __init__(
                self,
                model
                ):

        super().__init__()
        self.model = model

    def _sigma_inv(self, sigma):
        return 2. * torch.arccos(1. / (1. + sigma ** 2.).sqrt()) / np.pi

    def forward(self, x, sigma, y=None):
        x = x.to(torch.float32)
        c_skip = 1. / (sigma ** 2. + 1.)
        c_out = sigma / torch.sqrt(1. + sigma ** 2.)
        c_in = 1. / torch.sqrt(1. + sigma ** 2.)
        c_noise = self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised
    

class VESDEDenoiser(nn.Module):
    def __init__(self,
                sigma_min,
                sigma_max,
                model,
                ):

        super().__init__()

        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, x, sigma, y=None):
        
        x = x.to(torch.float32)

        c_skip = 1. 
        ### Essential adjustment for mimic data
        # c_skip = 0.11 ** 2. / \
        #     (sigma ** 2. + 0.11 ** 2.)
        
        c_out = sigma
        c_in = 1.
        c_noise = torch.log(sigma / 2.)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised

    

class VPSDEDenoiser(nn.Module):
    def __init__(
                self,
                beta_min,
                beta_d,
                M,
                eps_t,
                model
                ):

        super().__init__()

        self.model = model
        self.M = M
        self.beta_min = beta_min
        self.beta_d = beta_d
        ### https://github.com/NVlabs/edm/blob/main/training/networks.py
        self.sigma_min = float(self.sigma(eps_t))
        self.sigma_max = float(self.sigma(1))

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()
    
    def _sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d

    def forward(self, x, sigma, y=None):

        x = x.to(torch.float32)
        
        c_skip = 1.
        ### Essential adjustment for mimic data
        # c_skip = 0.13 ** 2. / \
        #     (sigma ** 2. + 0.13 ** 2.)
        
        c_out = -sigma
        c_in = 1. / torch.sqrt(sigma ** 2. + 1.)
        c_noise = (self.M-1) * self._sigma_inv(sigma)

        out = self.model(c_in * x, c_noise.reshape(-1), y)
        x_denoised = c_skip * x + c_out * out
        return x_denoised
