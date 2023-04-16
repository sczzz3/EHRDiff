from tqdm import tqdm
import torch
import numpy as np

from diffusion_util import LinearModel, Diffusion

device = torch.device('cuda:0')

dm = LinearModel(z_dim=1782, time_dim=384, unit_dims=[1024, 384, 384, 384, 1024])
dm.load_state_dict(torch.load("weight/model.pt"))
dm.to(device)

diffusion = Diffusion(
                        dm, 
                        dim = 1782,
                        P_mean = -1.2,          
                        P_std = 1.2,   
                        sigma_data = 0.14,  

                        num_sample_steps = 32, 
                        sigma_min = 0.02,      
                        sigma_max = 80,            
                        rho = 7,                    
                        )

out = []
dm.eval()
for b in tqdm(range(41), desc='Sampling...'):
    sampled_seq = diffusion.sample(batch_size=1000)
    out.append(sampled_seq)
out_seq = torch.cat(out)
out_seq = out_seq.detach().cpu().numpy()
res = np.rint(np.clip(out_seq, 0, 1))
np.save("EHRDiff", out_seq)


