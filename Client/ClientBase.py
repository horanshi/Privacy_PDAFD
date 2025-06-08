import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict

class Client(object):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, train_loader, test_loader, idx, logger, device):
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.idx = idx
        self.ce = nn.CrossEntropyLoss() 
        self.device = device
        self.kld = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        self.model = copy.deepcopy(model)

    def test_accuracy(self):
        self.model.eval()
        total_miss = 0
        total_leng = 0
        for batch_idx, (X, y) in enumerate(self.test_loader):  # batch_idx,
            X = X.to(self.device)
            y = y.to(self.device)
            out = self.model(X)
            y_pred = out.argmax(1)
            miss, leng = Accuracy(y,y_pred)
            total_miss += miss
            total_leng += leng
        return (total_leng-total_miss) / total_leng

    def train_accuracy(self):
        self.model.eval()
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.test_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            out = self.model(X)
            y_pred = out.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt

    def load_model(self,global_weights):
        self.model.load_state_dict(global_weights)