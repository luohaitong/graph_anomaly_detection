#!/home/zhengyu/anaconda/bin/python
#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error

import numpy as np
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader

import os

import config.const as const_util


class TestDataset(Dataset):

    def __init__(self, rm):

        self.rm = rm
    
    def __getitem__(self, index):

        items = self.rm.item_id[index]
        cats = self.rm.cats[index]
        PA_level = self.rm.PA_level[index]
        features = self.rm.features[index]

        '''
        items = torch.LongTensor(items)
        cats = torch.LongTensor(cats)
        PA_level = torch.LongTensor(PA_level)
        features = torch.FloatTensor(features)
        '''


        return items, cats, PA_level, features

    def __len__(self):

        return self.rm.item_id.size(0)


class FactorizationDataset(Dataset):

    def __init__(self, item_id, cat_r, cat_n, PA_level):

        self.item_id = item_id
        self.cat_r = cat_r
        self.cat_n = cat_n
        self.PA_level = PA_level

    def __getitem__(self, index):

        return self.item_id[index], self.cat_r[index], self.cat_n[index], self.PA_level[index]

    def __len__(self):

        return self.cat_r.size(0)


class PriceLevelFactorizationDataLoaderGenerator(object):

    def __init__(self, datafile_prefix):

        self.datafile_prefix = datafile_prefix
    
    def generate(self, batch_size, num_workers):

        self.sample_path = os.path.join(self.datafile_prefix, const_util.train_data)

        self.load_data()

        self.make_dataset()

        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    
    def load_data(self):

        self.item_id = torch.load(os.path.join(self.sample_path, const_util.item_id))
        self.cats_r = torch.load(os.path.join(self.sample_path, const_util.cats_r))
        self.cats_n = torch.load(os.path.join(self.sample_path, const_util.cats_n))
        self.PA_level = torch.load(os.path.join(self.sample_path, const_util.PA_level))
    
    def make_dataset(self):

        self.dataset = FactorizationDataset(self.item_id, self.cats_r, self.cats_n, self.PA_level)


class PairwiseDataset(Dataset):

    def __init__(self, item_id, cat, PA_level):
        self.item_id = item_id
        self.cat = cat
        self.PA_level = PA_level

    def __getitem__(self, index):
        return self.item_id[index], self.cat[index], self.PA_level[index]

    def __len__(self):
        return self.cat.size(0)


class PriceLevelPairwiseDataLoaderGenerator(object):

    def __init__(self, datafile_prefix):
        self.datafile_prefix = datafile_prefix

    def generate(self, batch_size, num_workers):
        self.sample_path = os.path.join(self.datafile_prefix, 'train_data')

        self.load_data()

        self.make_dataset()

        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    def load_data(self):
        self.item_id = torch.load(os.path.join(self.sample_path, const_util.item_id))
        self.cats = torch.load(os.path.join(self.sample_path, const_util.cats))
        self.PA_level = torch.load(os.path.join(self.sample_path, const_util.PA_level))

    def make_dataset(self):
        self.dataset = PairwiseDataset(self.item_id, self.cats, self.PA_level)