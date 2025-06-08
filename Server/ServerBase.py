from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy


class Server(object):
    def __init__(self, args, global_model, train_loader, test_loader, logger, device):
        self.global_model = global_model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger
        self.device = device
        self.LocalModels = []
        
    def global_test_accuracy(self):
        self.global_model.eval()
        accuracy = 0
        cnt = 0
        for batch_idx, (X, y) in enumerate(self.test_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            out = self.global_model(X)
            y_pred = out.argmax(1)
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        return accuracy/cnt

    def Save_CheckPoint(self, save_path):
        torch.save(self.global_model.state_dict(), save_path)
    
