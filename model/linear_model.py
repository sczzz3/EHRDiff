
import math
import torch
import torch.nn as nn
from einops import rearrange


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
            z_dim=1782, 
            time_dim=384,
            unit_dims=[1024, 384, 384, 384, 1024],

            random_fourier_features=False,
            learned_sinusoidal_dim=32,

            use_cfg=False,
            num_classes=2,
            class_dim=128,
            ):
        super().__init__()
        
        num_linears = len(unit_dims)

        if random_fourier_features:
            self.time_embedding = nn.Sequential(
                RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, is_random=True),
                nn.Linear(learned_sinusoidal_dim+1, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            self.time_embedding = nn.Sequential(
                    SinusoidalPositionEmbeddings(z_dim),
                    nn.Linear(z_dim, time_dim),
                    nn.SiLU(),
                    nn.Linear(time_dim, time_dim),
                )

        self.block_in = Block(dim_in=z_dim, dim_out=unit_dims[0], time_emb_dim=time_dim)
        self.block_mid = nn.ModuleList()
        for i in range(num_linears-1):
            self.block_mid.append(Block(dim_in=unit_dims[i], dim_out=unit_dims[i+1], time_emb_dim=time_dim))
        self.block_out = Block(dim_in=unit_dims[-1], dim_out=z_dim, time_emb_dim=time_dim)

        ### Classifier-free 
        self.label_dim = num_classes
        self.use_cfg = use_cfg
        if use_cfg:
            self.class_emb = nn.Embedding(self.label_dim if not use_cfg else self.label_dim + 1, class_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(class_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )    

    def forward(self, x, time_steps, labels=None):
        
        time_steps = time_steps.float()
        t_emb = self.time_embedding(time_steps)
        if self.use_cfg:
            class_emb = self.class_mlp(self.class_emb(labels))
            t_emb += class_emb 

        x = self.block_in(x, t_emb)

        num_mid_blocks = len(self.block_mid)
        if num_mid_blocks > 0:
            for block in self.block_mid:
                x = block(x, t_emb)

        x = self.block_out(x, t_emb)
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


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
    
