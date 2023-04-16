import os
import time
import random
import logging

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from diffusion_util import LinearModel, Diffusion


def set_seed(seed=3407):  
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def train_diff(args):

    logging.info("Loading Data...")
    
    raw_data = np.load(args.data_file)

    class EHRDataset(torch.utils.data.Dataset):
        def __init__(self, data=raw_data):
            super().__init__()
            self.data = data

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, index: int):
            return self.data[index]

    dataset = EHRDataset(raw_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.if_shuffle, drop_last=args.if_drop_last)
    device = args.device

    model = LinearModel(z_dim=args.ehr_dim, time_dim=args.time_dim, unit_dims=args.mlp_dims)
    model.to(args.device)

    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}  ])
    if args.if_drop_last:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,\
                                num_training_steps=(raw_data.shape[0]//args.batch_size)*args.num_epochs)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,\
                                num_training_steps=(raw_data.shape[0]//args.batch_size+1)*args.num_epochs)

    diffusion = Diffusion(
                    model, 
                    num_sample_steps = args.num_sample_steps,
                    dim = args.ehr_dim,
                    sigma_min = args.sigma_min,      
                    sigma_max = args.sigma_max,      
                    sigma_data = args.sigma_data,  
                    rho = args.rho,              
                    P_mean = args.p_mean,       
                    P_std = args.p_std,           
                    )


    # timestamp = time.strftime("%m_%d_%H_%M", time.localtime())

    logging.info("Training...")
    
    
    train_dm_loss = 0
    train_cnt = 0
    train_steps = 0
    best_corr = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):

            optimizer.zero_grad()

            batch_size = batch.shape[0]
            batch = batch.to(device)
            
            loss_dm = diffusion(batch)
            train_dm_loss += loss_dm.item()
            train_cnt += batch_size

            train_steps += 1
            if train_steps % args.check_steps == 0:
                logging.info('[%d, %5d] dm_loss: %.10f' % (epoch+1, train_steps, train_dm_loss / train_cnt))

                model.eval()   
                if args.eval_samples < args.batch_size:
                    syn_data = diffusion.sample(batch_size=args.eval_samples).detach().cpu().numpy()
                else:
                    num_iters = args.eval_samples // args.batch_size
                    num_left = args.eval_samples % args.batch_size
                    syn_data = []
                    for _ in range(num_iters):
                        syn_data.append(diffusion.sample(batch_size=args.batch_size).detach().cpu().numpy())
                    syn_data.append(diffusion.sample(batch_size=num_left).detach().cpu().numpy())
                    syn_data = np.concatenate(syn_data)
                
                syn_data = np.rint(np.clip(syn_data, 0, 1))
                corr, nzc, flag = plot_dim_dist(raw_data, syn_data, args.model_setting, best_corr)
                logging.info('corr: %.4f, none-zero columns: %d'%(corr, nzc)) 

                if flag:
                    best_corr = corr
                #     checkpoints_dirname = timestamp + '_' + args.model_setting
                #     os.makedirs(checkpoints_dirname, exist_ok=True)
                #     torch.save(model.state_dict(), checkpoints_dirname + "/model")
                #     torch.save(optimizer.state_dict(), checkpoints_dirname + "/optim")
                    torch.save(model.state_dict(), 'weight/model.pt')
                    torch.save(optimizer.state_dict(), 'weight/optim.pt')
                    logging.info("New Weight saved!")
                
                logging.info("**************************************")
                model.train()

            loss_dm.backward()
            optimizer.step()
            scheduler.step()


def plot_dim_dist(train_data, syn_data, model_setting, best_corr):

    train_data_mean = np.mean(train_data, axis = 0)
    temp_data_mean = np.mean(syn_data, axis = 0)
    corr = pearsonr(temp_data_mean, train_data_mean)
    nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
    fitted_values = [slope * i + intercept for i in train_data_mean]
    identity_values = [1 * i + 0 for i in train_data_mean]

    ax.plot(train_data_mean, fitted_values, 'b', alpha=0.5)
    ax.plot(train_data_mean, identity_values, 'r', alpha=0.5)
    ax.scatter(train_data_mean, temp_data_mean, alpha=0.3)
    ax.set_title('corr: %.4f, none-zero columns: %d, slope: %.4f'%(corr[0], nzc, slope))
    ax.set_xlabel('Feature prevalence of real data')
    ax.set_ylabel('Feature prevalence of synthetic data')

    # fig.savefig('figs/{}.png'.format('Current_' + model_setting))
    fig.savefig('figs/{}.png'.format('Cur_res'))

    flag = False
    if corr[0] > best_corr:
        best_corr = corr[0]
        flag = True
        # fig.savefig('figs/{}.png'.format('Best_' + model_setting))
        fig.savefig('figs/{}.png'.format('Best_res'))

    plt.close(fig)
    return corr[0], nzc, flag
