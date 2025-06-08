import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict
from Client.ClientBase import Client
import gc
import os


class ClientFedMD(Client):
    def __init__(self, args, model, train_loader, test_loader, pub_train_loader, idx, logger, device, shadow):
        super().__init__(args, model, train_loader, test_loader, idx, logger, device)
        self.pub_train_loader = pub_train_loader
        self.shadow = shadow

    def update_weights(self, global_round):
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        print("Train on public data...")
        # For public dataset training
        for iter in range(self.args.pre_ep):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.pub_train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                out = self.model(X)
                loss = self.ce(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    print('| Global Round: {} | Client: {} | Local Epoch : {} | [{}/{}]\tPub Loss: {:.6f}'.format(
                            global_round, self.idx, iter, batch_idx * len(X),
                            len(self.pub_train_loader.dataset), loss.item()))

        print("Train on private data...")
        for iter in range(self.args.pre_ep):
            for batch_idx, (X, y) in enumerate(self.train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                out = self.model(X)
                loss = self.ce(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    print('| Global Round: {} | Client: {} | Local Epoch : {} | [{}/{}]\tLoss: {:.6f}'.format(
                        global_round, self.idx, iter, batch_idx * len(X),
                        len(self.train_loader.dataset), loss.item()))

        model_save_path = f"../{self.args.alg}/{self.args.beta}/{self.args.dataset}/client_{self.idx}_{global_round}.pt"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_save_path)

    def update_weights_MD(self, knowledges, global_round):
        self.model.to(self.device)
        self.model.train()
        global_soft_prediction = torch.stack(knowledges)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.co_lr)

        # Digest
        print("Digest...")
        for iter in range(self.args.digest_ep):
            self.model.train()
            for idx, (X_pub, y_pub) in enumerate(self.pub_train_loader):
                X_pub = X_pub.to(self.device)
                out_pub = self.model(X_pub)
                # out_pub = F.softmax(out_pub / temp, dim=-1)
                optimizer.zero_grad()
                l1_loss = nn.L1Loss()
                loss = l1_loss(out_pub, global_soft_prediction[idx].to(self.device))
                loss.backward()
                optimizer.step()
                if idx % 50 == 0:
                    print(
                        '| Global Round : {} | Client: {} | Local Epoch : {} | [{}/{}]\t Loss: {:.6f}'.format(
                            global_round, self.idx, iter, idx * len(X_pub),
                            len(self.pub_train_loader.dataset), loss.item()))
        accuracy = self.test_accuracy()
        print("Digest test accuracy:", accuracy)

        # Revisit
        print("Revisit...")
        for iter in range(self.args.revisit_ep):
            for batch_idx, (X, y) in enumerate(self.train_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                out_pub = self.model(X)
                loss = self.ce(out_pub, y)
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    print('| Global Round : {} | Client: {} | Local Epoch : {} | [{}/{}]\tLoss: {:.6f}'.format(
                        global_round, self.idx, iter, batch_idx * len(X),
                        len(self.train_loader.dataset),loss.item()))

        if global_round in [0, 4, 9]:
            model_save_path = f"../{self.args.alg}/{self.args.beta}/{self.args.dataset}/client_{self.idx}_{global_round}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(self.model.state_dict(), model_save_path)
    
    def generate_knowledge(self, temp):
        self.model.to(self.device)
        self.model.eval()
        num_classes = self.model.num_classes
        soft_predictions = []  #[[batch]]
        for batch_idx, (X, y) in enumerate(self.pub_train_loader):
            X = X.to(self.device)
            y = y
            out = self.model(X)
            # out = F.softmax(out/temp, dim=-1)
            Q = out.to(self.device).detach().cpu()
            soft_predictions.append(Q)
            del X
            del y
            del out
            del Q
            gc.collect()
         
        return soft_predictions

    def generate_train_data(self):
        self.model.to(self.device)
        self.model.eval()
        num_classes = self.model.num_classes
        soft_predictions = []
        soft_label = []
        for batch_idx, (X, y) in enumerate(self.pub_train_loader):
            X = X.to(self.device)
            y = y
            out = self.model(X)
            out = F.softmax(out, dim=-1)
            Q = out.to(self.device).detach().cpu()
            for predict_results in Q:
                soft_predictions.append(predict_results)
            for label in y:
                soft_label.append(label)
            del X
            del y
            del out
            del Q
            gc.collect()

        soft_predictions = torch.stack(soft_predictions)
        soft_label = torch.stack(soft_label)

        return soft_predictions, soft_label