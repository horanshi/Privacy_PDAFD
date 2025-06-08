from torch.utils.data import Dataset
import torch
import copy
from utils import Accuracy
from Server.ServerBase import Server
from Client.ClientFedMD import ClientFedMD
from tqdm import tqdm
import numpy as np
from utils import average_weights
import time
from sampling import LocalDataset, LocalDataloaders, partition_data
import gc
import random
import torch.nn.functional as F


class ServerFedMD(Server):
    def __init__(self, args, global_model, train_loader, test_loader, pub_train, logger, device):
        super().__init__(args, global_model, train_loader, test_loader, logger, device)
        self.public_train_loader = pub_train
    
    def Create_Clints(self):
        # Load the Clients
        for idx in range(self.args.num_clients):
            self.LocalModels.append(ClientFedMD(self.args, copy.deepcopy(self.global_model), self.train_loader[idx],
                                                self.test_loader[idx], self.public_train_loader, idx, self.logger,
                                                self.device, shadow=0))

    def train(self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        rng = np.random.default_rng(self.args.seed)

        start_time = time.time()
        test_data_attack_model = []  # [[tensor(tensor, tensor, ...), tensor(tensor, tensor, ...), ...], ...]
        test_data_attack_model_label = []
        for epoch in tqdm(range(self.args.num_rounds)):
            Knowledges = []
            Client_Adversarial_Knowledges = []  # [tensor(tensor, tensor, ...), tensor(tensor, tensor, ...), ...] [client_id][public_id] Store the knowledge for each client model
            Client_Adversarial_Labels = []
            print(f'\n | Global Training Round : {epoch+1} |\n')

            # select the client to join this round
            m = max(int(self.args.sampling_rate * self.args.num_clients), 1)
            idxs_users = rng.choice(range(self.args.num_clients), m, replace=False)
            total_idxs_users = idxs_users
            total_idxs_users = np.sort(total_idxs_users)
            print(total_idxs_users)

            # assign the public dataset
            if epoch < 1:
                dict_pub = [rng.choice(a=len(self.public_train_loader), size=5000, replace=False)]  # Set the size of public data set

            else:
                dict_pub = [rng.choice(a=len(self.public_train_loader), size=5000, replace=False)]  # Set the size of public data set

                # Communicate
                for idx in total_idxs_users:
                    self.LocalModels[idx].pub_train_loader = LocalDataloaders(self.public_train_loader, dict_pub, self.args.digest_batch_size, ShuffleorNot=False)[0]
                    knowledges = self.LocalModels[idx].generate_knowledge(temp=self.args.temp)
                    Knowledges.append(torch.stack(knowledges))
                    # Collect the logit and label for LDIA
                    Adversarial_Knowledges, Adversarial_Labels = self.LocalModels[idx].generate_train_data()
                    Client_Adversarial_Knowledges.append(Adversarial_Knowledges)  # collect logit [clients * probability]  # [client * [data * probability]]
                    Client_Adversarial_Labels.append(Adversarial_Labels)  # collect label [clients * labels]
                Client_Adversarial_Knowledges = torch.stack(Client_Adversarial_Knowledges)
                Client_Adversarial_Labels = torch.stack(Client_Adversarial_Labels)
                test_data_attack_model.append(Client_Adversarial_Knowledges)  # [epochs * clients * probability]  # [epochs * [client * [data * probability] ]]
                test_data_attack_model_label.append(Client_Adversarial_Labels)  # [epochs * [clients * labels]]

                # Aggregate the logit from all clients
                global_soft_prediciton = []
                batch_pub = Knowledges[0].shape[0]
                for i in range(batch_pub):
                    num = Knowledges[0].shape[1]
                    soft_label = torch.zeros(num, self.args.num_classes)  # tensor(batch_size * classes)
                    for idx in idxs_users:
                        soft_label += Knowledges[idx][i]
                    soft_label = soft_label / len(idxs_users)
                    global_soft_prediciton.append(soft_label)  # [tensor(batch_size * classes),]

            for idx in total_idxs_users:
                if epoch < 1:
                    # Transfer Learning
                    self.LocalModels[idx].pub_train_loader = LocalDataloaders(self.public_train_loader, dict_pub, self.args.batch_size, ShuffleorNot=False)[0]
                    self.LocalModels[idx].update_weights(global_round=epoch)
                    acc = self.LocalModels[idx].test_accuracy()
                    print("Local Training Test Accuracy:", acc)
                    
                else:
                    # Distribute, Digest & Revisit
                    self.LocalModels[idx].update_weights_MD(knowledges=global_soft_prediciton, global_round=epoch)
                    acc = self.LocalModels[idx].test_accuracy()
                    print("Revisit Test Accuracy:", acc)

        print('Training is completed.')
        end_time = time.time()
        print('running time: {} s '.format(end_time - start_time))

        test_data_attack_model = torch.stack(test_data_attack_model)
        test_data_attack_model_label = torch.stack(test_data_attack_model_label)

        return test_data_attack_model, test_data_attack_model_label