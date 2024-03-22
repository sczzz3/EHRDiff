import os
import logging
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from model.linear_model import LinearModel
from utils.util import set_seeds, make_dir, save_checkpoint, sample_random_batch, plot_dim_dist
from model.ema import ExponentialMovingAverage
from score_losses import EDMLoss, VPSDELoss, VESDELoss#, VLoss
from denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, NaiveDenoiser#, VDenoiser
from samplers import ablation_sampler#, ddim_sampler, edm_sampler

import importlib
opacus = importlib.import_module('src.opacus')

from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from transformers import get_cosine_schedule_with_warmup


def training(config, workdir, mode):
    
    set_seeds(config.setup.global_rank, config.train.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples')
    checkpoint_dir = os.path.join(workdir, 'checkpoints')

    if config.setup.global_rank == 0:
        if mode == 'train':
            make_dir(sample_dir)
            make_dir(checkpoint_dir)
    dist.barrier()

    if config.model.denoiser_name == 'edm':
        if config.model.denoiser_network == 'song':
            model = EDMDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
            # model = NaiveDenoiser(
            #     model=LinearModel(**config.model.network).to(config.setup.device))          
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vpsde':
        if config.model.denoiser_network == 'song':
            model = VPSDEDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'vesde':
        if config.model.denoiser_network == 'song':
            model = VESDEDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
        else:
            raise NotImplementedError
    elif config.model.denoiser_name == 'naive':
            model = NaiveDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device))
    # elif config.model.denoiser_name == 'v':
    #     if config.model.denoiser_network == 'song':
    #         model = VDenoiser(
    #             model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
    #     else:
    #         raise NotImplementedError
    else:
        raise NotImplementedError


    if config.dp.do:
        model = DPDDP(model)
    else:
        model = DistributedDataParallel(model.to(config.setup.device), device_ids=[config.setup.device])

    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)

    if config.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), **config.optim.params)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), **config.optim.params)
    else:
        raise NotImplementedError

    raw_data = np.load(config.data.path)
    if not config.dp.do:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.train.warmup_steps, \
                                num_training_steps=(raw_data.shape[0]//config.train.batch_size+1)*config.train.n_epochs)
    

    state = dict(model=model, ema=ema, optimizer=optimizer, step=0)

    if config.setup.global_rank == 0:
        model_parameters = filter(
            lambda p: p.requires_grad, model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info('Number of trainable parameters in model: %d' % n_params)
        logging.info('Number of total epochs: %d' % config.train.n_epochs)
        logging.info('Starting training at step %d' % state['step'])
    dist.barrier()
        
    if config.data.name.startswith('mimic'):
        labels = None
        EHR_task = 'binary'
    elif config.data.name.startswith('cinc') or config.data.name.startswith('ecg'):
        EHR_task = 'continuous'
        if config.model.network.use_cfg:
            raw_data = raw_data[:, 1:]
            labels = raw_data[:, 0]
        else:
            labels = None

    class EHRDataset(torch.utils.data.Dataset):
        def __init__(self, data=raw_data, labels=None):
            super().__init__()
            self.data = data
            self.labels = labels

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, index: int):
            if self.labels is not None:
                return self.data[index], self.labels[index]
            else:
                return self.data[index]

    dataset = EHRDataset(raw_data, labels)
    dataset_loader = torch.utils.data.DataLoader(
        dataset=dataset, shuffle=True, batch_size=config.train.batch_size)

    if config.dp.do:
        privacy_engine = PrivacyEngine()

        # model, optimizer, dataset_loader = privacy_engine.make_private(
        model, optimizer, dataset_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataset_loader,
            # noise_multiplier=.7,
            target_delta=config.dp.delta,
            target_epsilon=config.dp.epsilon,
            epochs=config.train.n_epochs,
            max_grad_norm=config.dp.max_grad_norm,
            noise_multiplicity=config.loss.n_noise_samples,
        )
        

    if config.loss.n_classes == 'None':
        config.loss.n_classes = None
    if config.loss.version == 'edm':
        loss_fn = EDMLoss(**config.loss).get_loss
    elif config.loss.version == 'vpsde':
        loss_fn = VPSDELoss(**config.loss).get_loss
    elif config.loss.version == 'vesde':
        loss_fn = VESDELoss(**config.loss).get_loss
    # elif config.loss.version == 'v':
    #     loss_fn = VLoss(**config.loss).get_loss
    else:
        raise NotImplementedError


    if config.sampler.guid_scale == 'None':
        config.sampler.guid_scale = None
    def sampler(x, y=None):
        # if config.sampler.type == 'ddim':
        #     return ddim_sampler(x, y, model, **config.sampler)
        # elif config.sampler.type == 'edm':
        #     return edm_sampler(x, y, model, **config.sampler)
        # else:
        #     raise NotImplementedError
        return ablation_sampler(x, y, model, **config.sampler)


    # snapshot_sampling_shape = (config.sampler.snapshot_batch_size, config.data.resolution)
    snapshot_sampling_shape = (raw_data.shape[0], config.data.resolution)
    if config.data.n_classes == 'None':
        config.data.n_classes = None

    for epoch in range(config.train.n_epochs):
        if config.dp.do:
            with BatchMemoryManager(
                    data_loader=dataset_loader,
                    max_physical_batch_size=config.dp.max_physical_batch_size,
                    optimizer=optimizer,
                    n_splits=config.dp.n_splits if config.dp.n_splits > 0 else None) as memory_safe_data_loader:

                for _, (train) in enumerate(memory_safe_data_loader):

                    if isinstance(train, list):
                        train_x = train[0]
                        train_y = train[1]
                    else:
                        train_x = train
                        train_y = None

                    x = train_x.to(config.setup.device).to(torch.float32)

                    if config.data.n_classes is None:
                        y = None
                    else:
                        y = train_y.to(config.setup.device)
                        if y.dtype == torch.float32:
                            y = y.long()

                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean(loss_fn(model, x, y))
                    loss.backward()
                    optimizer.step()

                    if state['step'] % config.train.check_freq == 0 and state['step'] >= config.train.check_freq and config.setup.global_rank == 0:
                        model.eval()
                        with torch.no_grad():
                            ema.store(model.parameters())
                            ema.copy_to(model.parameters())
                            if config.setup.local_rank == 0:
                                sample_random_batch(EHR_task, snapshot_sampling_shape, sampler, 
                                                        sample_dir, config.setup.device, config.data.n_classes)
                            torch.cuda.empty_cache()
                            ema.restore(model.parameters())
                        model.train()

                        logging.info('[%d, %5d] Loss: %.10f' % (epoch+1, state['step'] + 1, loss))
                        syn_data = np.load(sample_dir + '/sample.npy')
                        corr, nzc = plot_dim_dist(raw_data, syn_data, workdir)
                        logging.info('corr: %.4f, none-zero columns: %d'%(corr, nzc)) 
                        logging.info('Eps-value: %.4f' % (privacy_engine.get_epsilon(config.dp.delta)))
                    dist.barrier()


                    if state['step'] % config.train.save_freq == 0 and state['step'] >= config.train.save_freq and config.setup.global_rank == 0:
                        checkpoint_file = os.path.join(
                            checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                        save_checkpoint(checkpoint_file, state)
                        logging.info(
                            'Saving checkpoint at iteration %d' % state['step'])
                        logging.info('--------------------------------------------')
                    dist.barrier()

                    state['step'] += 1
                    if not optimizer._is_last_step_skipped:
                        state['ema'].update(model.parameters())


        else: # with No Differential Private training
            for _, (train) in enumerate(dataset_loader):

                if isinstance(train, list):
                    train_x = train[0]
                    train_y = train[1]
                else:
                    train_x = train
                    train_y = None

                x = train_x.to(config.setup.device).to(torch.float32)

                if config.data.n_classes is None:
                    y = None
                else:
                    y = train_y.to(config.setup.device)
                    if y.dtype == torch.float32:
                        y = y.long()

                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean(loss_fn(model, x, y))
                # if config.setup.local_rank == 0:
                #     print(loss)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if state['step'] % config.train.check_freq == 0 and state['step'] >= config.train.check_freq and config.setup.local_rank == 0:
                    model.eval()
                    with torch.no_grad():
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        if config.setup.local_rank == 0:
                            sample_random_batch(EHR_task, snapshot_sampling_shape, sampler, 
                                                    sample_dir, config.setup.device, config.data.n_classes)
                        torch.cuda.empty_cache()
                        ema.restore(model.parameters())
                    model.train()

                    logging.info('[%d, %5d] Loss: %.10f' % (epoch+1, state['step'] + 1, loss))
                    syn_data = np.load(sample_dir + '/sample.npy')
                    corr, nzc = plot_dim_dist(raw_data, syn_data, workdir)
                    logging.info('corr: %.4f, none-zero columns: %d'%(corr, nzc)) 
                dist.barrier()

                if state['step'] % config.train.save_freq == 0 and state['step'] >= config.train.save_freq and config.setup.local_rank == 0:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 'checkpoint_%d.pth' % state['step'])
                    save_checkpoint(checkpoint_file, state)
                    logging.info(
                        'Saving checkpoint at iteration %d' % state['step'])
                    logging.info('--------------------------------------------')
                dist.barrier()

                state['step'] += 1
                state['ema'].update(model.parameters())

        
    if config.setup.local_rank == 0:
        checkpoint_file = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
        save_checkpoint(checkpoint_file, state)
        logging.info('Saving final checkpoint.')
    dist.barrier()

    model.eval()
    with torch.no_grad():
        ema.store(model.parameters())
        ema.copy_to(model.parameters())

        if config.setup.local_rank == 0:
            logging.info('################################################')
            logging.info('Final Evaluation')
            syn_data = np.load(sample_dir + '/sample.npy')
            corr, nzc = plot_dim_dist(raw_data, syn_data, workdir)
            logging.info('corr: %.4f, none-zero columns: %d'%(corr, nzc)) 
        dist.barrier()

        ema.restore(model.parameters())
