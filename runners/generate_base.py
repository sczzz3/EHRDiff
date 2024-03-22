import torch
import logging
import torch.distributed as dist
import numpy as np
import os
from torch.nn.parallel import DistributedDataParallel as DDP

from model.ema import ExponentialMovingAverage
from utils.util import set_seeds, make_dir
from denoiser import EDMDenoiser, VPSDEDenoiser, VESDEDenoiser, NaiveDenoiser#, VDenoiser
from samplers import ablation_sampler# ddim_sampler, edm_sampler
from model.linear_model import LinearModel


def get_model(config, local_rank):
    if config.model.denoiser_name == 'edm':
        if config.model.denoiser_network == 'song':
            model = EDMDenoiser(
                model=LinearModel(**config.model.network).to(config.setup.device), **config.model.params)
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
    #             LinearModel(**config.model.network).to(config.setup.device))
        # else:
        #     raise NotImplementedError
    else:
        raise NotImplementedError

    model = DDP(model, device_ids=[local_rank])
    state = torch.load(config.model.ckpt, map_location=config.setup.device)
    logging.info(model.load_state_dict(state['model'], strict=True))
    if config.model.use_ema:
        ema = ExponentialMovingAverage(
            model.parameters(), decay=config.model.ema_rate)
        ema.load_state_dict(state['ema'])
        ema.copy_to(model.parameters())

    model.eval()
    return model


def sample_batch(counter, max_samples, sampling_fn, sampling_shape, device, labels, n_classes):
    x = torch.randn(sampling_shape, device=device)
    with torch.no_grad():
        if labels is None:
            if n_classes is not None:
                raise ValueError(
                    'Need to set labels for class-conditional sampling.')

            x = sampling_fn(x)
        else:
            if isinstance(labels, int):
                if labels == n_classes:
                    labels = torch.randint(
                        n_classes, (sampling_shape[0],)).to(x.device)
                else:
                    labels = torch.tensor(
                        sampling_shape[0] * [labels], device=x.device)
            else:
                raise NotImplementedError

            x = sampling_fn(x, labels)    

    for _ in x:
        if counter < max_samples:
            counter += 1

    return counter, x, labels


def evaluation(config, workdir):
    if config.model.ckpt is None:
        raise ValueError('Need to specify a checkpoint.')

    set_seeds(config.setup.global_rank, config.test.seed)
    torch.cuda.device(config.setup.local_rank)
    config.setup.device = 'cuda:%d' % config.setup.local_rank

    sample_dir = os.path.join(workdir, 'samples/')
    if config.setup.global_rank == 0:
        make_dir(sample_dir)
    dist.barrier()

    model = get_model(config, config.setup.local_rank)
    
    sampling_shape = (config.test.batch_size, config.data.resolution)
    if config.data.n_classes == 'None':
        config.data.n_classes = None
        
    if config.sampler.guid_scale == 'None':
        config.sampler.guid_scale = None
    if config.test.labels == 'None':
        config.test.labels = None
    def sampler(x, y=None):
        # if config.sampler.type == 'ddim':
        #     return ddim_sampler(x, y, model, **config.sampler)
        # elif config.sampler.type == 'edm':
        #     return edm_sampler(x, y, model, **config.sampler)
        # else:
        #     raise NotImplementedError
        return ablation_sampler(x, y, model, **config.sampler)

    counter = (config.test.n_samples //
               (sampling_shape[0] * config.setup.global_size) + 1) * sampling_shape[0] * config.setup.global_rank


    all_labels = []
    all_x = []
    # for _ in range(config.test.n_samples // (sampling_shape[0] * config.setup.global_size) + 1):
    #     counter, x, labels = sample_batch(counter, config.test.n_samples, sampler,
    #                                    sampling_shape, config.setup.device, config.test.labels, config.data.n_classes)
    #     all_labels.append(labels)
    #     all_x.append(x)
    counter, x, labels = sample_batch(counter, config.test.n_samples, sampler,
                                    sampling_shape, config.setup.device, config.test.labels, config.data.n_classes)
    all_labels.append(labels)
    all_x.append(x)    

    if config.test.labels is not None:
        all_labels = torch.cat(all_labels)
        all_x = torch.cat(all_x)

        all_labels_across_all_gpus = [torch.empty_like(all_labels).to(
            config.setup.device) for _ in range(config.setup.global_size)]
        all_x_across_all_gpus = [torch.empty_like(all_x).to(
            config.setup.device) for _ in range(config.setup.global_size)]
        
        dist.all_gather(all_labels_across_all_gpus, all_labels)
        dist.all_gather(all_x_across_all_gpus, all_x)

        all_labels_across_all_gpus = torch.cat(all_labels_across_all_gpus)[
            :config.test.n_samples].to('cpu')
        all_x_across_all_gpus = torch.cat(all_x_across_all_gpus)[
            :config.test.n_samples].to('cpu')
        
        if config.setup.global_rank == 0:
            # torch.save(all_labels_across_all_gpus,
            #            os.path.join(sample_dir, 'all_labels.pt'))
            all_x_across_all_gpus = all_x_across_all_gpus.numpy()
            if config.test.type == 'binary':
                all_x_across_all_gpus = np.rint(np.clip(all_x_across_all_gpus, 0, 1))
            elif config.test.type == 'continuous':
                all_x_across_all_gpus = np.clip(all_x_across_all_gpus, 0, 1)

            np.save(os.path.join(sample_dir, 'all_labels'),
                    all_labels_across_all_gpus)
            np.save(os.path.join(sample_dir, 'all_x'),
                    all_x_across_all_gpus)
        dist.barrier()
    
    else:
        all_x = torch.cat(all_x)

        all_x_across_all_gpus = [torch.empty_like(all_x).to(
            config.setup.device) for _ in range(config.setup.global_size)]
        
        dist.all_gather(all_x_across_all_gpus, all_x)

        all_x_across_all_gpus = torch.cat(all_x_across_all_gpus)[
            :config.test.n_samples].to('cpu')
        
        if config.setup.global_rank == 0:
            all_x_across_all_gpus = all_x_across_all_gpus.numpy()
            if config.test.type == 'binary':
                all_x_across_all_gpus = np.rint(np.clip(all_x_across_all_gpus, 0, 1))
            elif config.test.type == 'continuous':
                all_x_across_all_gpus = np.clip(all_x_across_all_gpus, 0, 1)

            np.save(os.path.join(sample_dir, 'all_x'),
                    all_x_across_all_gpus)
            
            ###
            # np.save('saved_skips', np.array(torch.as_tensor(model.module.skip).cpu()))
            ###
        dist.barrier()        
