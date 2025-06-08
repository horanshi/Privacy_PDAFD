import numpy as np
from torch.utils.data import Dataset, Subset
import torch
from torchvision import datasets, transforms
from collections import Counter
import random




class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """

    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y


def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot=True, BatchorNot=True):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        if BatchorNot == True:
            loader = torch.utils.data.DataLoader(LocalDataset(dataset, dict_users[i]),
                                                 batch_size=batch_size,
                                                 shuffle=ShuffleorNot,
                                                 num_workers=0,
                                                 drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(LocalDataset(dataset, dict_users[i]),
                                                 batch_size=len(LocalDataset(dataset, dict_users[i])),
                                                 shuffle=ShuffleorNot,
                                                 num_workers=0,
                                                 drop_last=True)
        loaders.append(loader)
    return loaders


def partition_data(n_users, alpha=0.5, rand_seed=0, dataset='cifar10', public_ratio=0.1):
    #set the seed
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    if dataset == 'CIFAR10':
        K = 10
        data_dir = '../data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        len_train = len(train_dataset)
        len_public = int(public_ratio * len_train)
        len_train = len_train - len_public
        train_dataset, public_dataset = torch.utils.data.random_split(train_dataset,[len_train, len_public])

        labels = [label for _, label in public_dataset]
        label_counts = Counter(labels)
        print(label_counts)

        train_indices = train_dataset.indices
        y_train = np.array([train_dataset.dataset.targets[i] for i in train_indices])
        y_test = np.array(test_dataset.targets)

    min_size = 0
    N = len(train_dataset)
    net_dataidx_map = {}
    net_dataidx_map_test = {}

    while min_size < 64:
        idx_batch = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]  # Get the index for specific class in training set
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            # balance
            proportions_train = np.array([p * (len(idx_j) < N / n_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions_train = proportions_train / proportions_train.sum()
            proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    for j in range(n_users):
        net_dataidx_map_test[j] = list(range(len(y_test)))

    return train_dataset, test_dataset, public_dataset, net_dataidx_map, net_dataidx_map_test  #  public_dataset, auxiliary_dataset,


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts


def generate_shadow_training_data(auxiliary_dataset, dataset, shadow_beta, shadow_users=2):
    if dataset == 'CIFAR10':
        K = 10
    N = len(auxiliary_dataset)
    net_dataidx_map = {}

    y_auxiliary = []
    for i in range(len(auxiliary_dataset)):
        _, target = auxiliary_dataset[i]
        y_auxiliary.append(target)
    y_auxiliary = np.array(y_auxiliary)

    for i in range(len(shadow_beta)):
        min_size = 0
        while min_size < 100:
            idx_batch = [[] for _ in range(shadow_users)]
            print('generate shadow training data alpha:{}'.format(shadow_beta[i]))
            for k in range(K):
                idx_k = np.where(y_auxiliary == k)[0]  # Get the index for specific class in training set
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(shadow_beta[i], shadow_users))
                # formatted_list = [f"{x:.2f}" for x in proportions]
                # print(formatted_list)
                ## Balance
                proportions_train = np.array([p * (len(idx_j) < N / shadow_users) for p, idx_j in zip(proportions, idx_batch)])
                proportions_train = proportions_train / proportions_train.sum()
                proportions_train = (np.cumsum(proportions_train) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions_train))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(shadow_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[i * shadow_users + j] = idx_batch[j]

    return net_dataidx_map
