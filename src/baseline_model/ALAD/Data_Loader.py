import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image


class dataset_loader(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        """number of samples."""
        return len(self.data)


def get_data(args, data_dir='../../data/DGraphFin/dgraphfin.npz'):
    """get dataloders"""

    dataset = np.load(data_dir)
    data = dataset['x']
    labels = dataset['y']

    normal_data = data[labels == args.normal_class]
    normal_labels = labels[labels == args.normal_class]
    anormal_data = data[labels == args.abnormal_class]
    anormal_labels = labels[labels == args.abnormal_class]

    N_train = int(normal_data.shape[0] * 0.8)

    x_train = normal_data[:N_train]
    y_train = normal_labels[:N_train]
    data_train = dataset_loader(x_train, y_train)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)

    x_test = np.concatenate((anormal_data, normal_data[N_train:]), axis=0)
    y_test = np.concatenate((anormal_labels, normal_labels[N_train:]), axis=0)
    y_test = np.where(y_test == args.normal_class, 0, 1)
    data_test = dataset_loader(x_test, y_test)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0)
    return dataloader_train, dataloader_test




