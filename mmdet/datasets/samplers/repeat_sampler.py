import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
import itertools


class RepeatFactorTrainingSampler(Sampler):
    def __init__(self, 
                 dataset, 
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 repeat_thresh=1.0, 
                 shuffle=True, 
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        self.num_samples = 0

        self.shuffle = shuffle
        

    def _get_epoch_indices(self, generator):
        indices = []
        return indices

    def __iter__(self):
        start = self.rank
        world_size = self.num_replicas
        yield from itertools.islice(self._infinite_indices(), start, None, world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self.shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch












