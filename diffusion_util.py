# -----------------------------------
# Code adapted from: 
# https://github.com/lucidrains/denoising-diffusion-pytorch
# -----------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None):
        super().__init__()

        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, dim_in),
            )
        
        self.out_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_in, dim_out),
        )

    def forward(self, x, time_emb=None):
        
        if time_emb is not None:
            t_emb = self.time_mlp(time_emb)
            h = x + t_emb  
        else:
            h = x
        out = self.out_proj(h)
        return out  


class LinearModel(nn.Module):
    def __init__(
            self, *,
            z_dim, 
            time_dim,
            unit_dims,
            ):
        super().__init__()
        
        num_linears = len(unit_dims)
        self.time_embedding = nn.Sequential(
                SinusoidalPositionEmbeddings(z_dim),
                nn.Linear(z_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )

        self.block_in = Block(dim_in=z_dim, dim_out=unit_dims[0], time_emb_dim=time_dim)
        self.block_mid = nn.ModuleList()
        for i in range(num_linears-1):
            self.block_mid.append(Block(dim_in=unit_dims[i], dim_out=unit_dims[i+1]))
        self.block_out = Block(dim_in=unit_dims[-1], dim_out=z_dim)

    def forward(self, x, time_steps):

        t_emb = self.time_embedding(time_steps)
        x = self.block_in(x, t_emb)

        num_mid_blocks = len(self.block_mid)
        if num_mid_blocks > 0:
            for block in self.block_mid:
                x = block(x)

        x = self.block_out(x)
        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    
class Diffusion(nn.Module):
    def __init__(
            self,
            net,
            *,
            dim,
            num_sample_steps, 
            sigma_min,     
            sigma_max,        
            sigma_data,     
            rho,              
            P_mean,        
            P_std,         
        ):
        super().__init__()

        self.net = net
        self.dim = dim

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  


    @property
    def device(self):
        return next(self.net.parameters()).device


    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25


    def preconditioned_network_forward(self, noised_ehr, sigma, clamp = False):
        batch, device = noised_ehr.shape[0], noised_ehr.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_ehr,
            self.c_noise(sigma),
        )
        
        out = self.c_skip(padded_sigma) * noised_ehr +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(0, 1)
        return out

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) 
        return sigmas


    @torch.no_grad()
    def sample(self, batch_size = 32, num_sample_steps = None, clamp = True):

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.dim)

        sigmas = self.sample_schedule(num_sample_steps)

        sigmas_and_sigmas_next = list(zip(sigmas[:-1], sigmas[1:]))

        init_sigma = sigmas[0]

        ehr = init_sigma * torch.randn(shape, device = self.device)

        for sigma, sigma_next in sigmas_and_sigmas_next:

            sigma, sigma_next = map(lambda t: t.item(), (sigma, sigma_next))

            model_output = self.preconditioned_network_forward(ehr, sigma, clamp = clamp)

            denoised_over_sigma = (ehr - model_output) / sigma

            ehr_next = ehr + (sigma_next - sigma) * denoised_over_sigma

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(ehr_next, sigma_next, clamp = clamp)

                denoised_prime_over_sigma = (ehr_next - model_output_next) / sigma_next
                ehr_next = ehr + 0.5 * (sigma_next - sigma) * (denoised_over_sigma + denoised_prime_over_sigma)

            ehr = ehr_next
        return ehr

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, ehr):
        
        batch_size = ehr.shape[0]
        sigmas = self.noise_distribution(batch_size)
        
        padded_sigmas = rearrange(sigmas, 'b -> b 1')

        noise = torch.randn_like(ehr)

        noised_ehr = ehr + padded_sigmas * noise

        denoised = self.preconditioned_network_forward(noised_ehr, sigmas)

        losses = F.mse_loss(denoised, ehr, reduction='none')
        losses = reduce(losses, 'b ... -> b', 'mean')
        losses = losses * self.loss_weight(sigmas)

        return losses.mean()
