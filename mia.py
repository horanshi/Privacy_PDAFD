import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, Dataset, SubsetRandomSampler
import torchvision.models as models
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch.distributions.normal import Normal
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import torch.optim as optim
import os
from collections import Counter
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--attack',  default='coop',help='coop, distill')
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--alg', default='FedMD')
    parser.add_argument('--epoch', type=int, default=0, help='epoch phase')
    parser.add_argument('--beta', type=float, default=0.1, help='beta for non-iid distribution')
    parser.add_argument('--num_shadows', type=int, default=8, help='numbers of shadow models')

    args = parser.parse_args()
    return args


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

        self.model.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.model.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.model.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.model.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

        self.model.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.model.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.model.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.model.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

        self.model.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.model.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.model.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.model.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

        if hasattr(self.model.layer2[0], 'downsample'):
            self.model.layer2[0].downsample[1] = nn.GroupNorm(num_groups=32, num_channels=128)
        if hasattr(self.model.layer3[0], 'downsample'):
            self.model.layer3[0].downsample[1] = nn.GroupNorm(num_groups=32, num_channels=256)
        if hasattr(self.model.layer4[0], 'downsample'):
            self.model.layer4[0].downsample[1] = nn.GroupNorm(num_groups=32, num_channels=512)

        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out


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
            loader = torch.utils.data.DataLoader(LocalDataset(dataset, dict_users[i]),batch_size=batch_size,shuffle=ShuffleorNot, num_workers=0,drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(LocalDataset(dataset, dict_users[i]),batch_size=len(LocalDataset(dataset, dict_users[i])),shuffle=ShuffleorNot,num_workers=0,drop_last=True)
        loaders.append(loader)
    return loaders


# Co-op LiRA
def get_mia_score_coop(data_loader, model_list, target):
    clients_score_list = []
    for i in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], desc="Inferencing models"):
        score_list = []
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model_list[i](inputs)
                outputs = F.softmax(outputs, dim=1)
                p = outputs.gather(1, labels.view(-1, 1)).squeeze(1)
                complement = 1 - p
                p = torch.clamp(p, min=epsilon, max=1 - epsilon)
                complement = torch.clamp(complement, min=epsilon)
                score = torch.log(p) - torch.log(complement)
            score_list.append(score)

        score_list = torch.cat(score_list, dim=0)
        clients_score_list.append(score_list)

    clients_score_list = torch.stack(clients_score_list)
    data_len = clients_score_list[0].size(0)
    
    data_prob_list = []
    for data in range(data_len):
        data_score = torch.cat([clients_score_list[:target, data], clients_score_list[target+1:, data]])
        mean = data_score.mean().item()
        std = data_score.std().item()
        if std == 0:
            std = epsilon
        normal_dist = Normal(mean, std)
        prob = normal_dist.cdf(clients_score_list[target, data]).item()
        data_prob_list.append(prob)

    return data_prob_list


# Distillation based-LiRA
def get_mia_score_distill(data_loader, model_list, shadow_model_list, target):
    clients_score_list = []
    target_score_list = []
    model_list[target].eval()
    for _, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model_list[target](inputs)
            outputs = F.softmax(outputs, dim=1)
            p = outputs.gather(1, labels.view(-1, 1)).squeeze(1)
            complement = 1 - p
            p = torch.clamp(p, min=epsilon, max=1 - epsilon)
            complement = torch.clamp(complement, min=epsilon)
            score = torch.log(p) - torch.log(complement)
        target_score_list.append(score)
    target_score_list = torch.cat(target_score_list, dim=0)
    clients_score_list.append(target_score_list)

    # get scores on shadow models
    for i in tqdm(range(args.num_shadows), desc="Inferencing shadow models"):
        score_list = []
        shadow_model_list[i].eval()
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = shadow_model_list[i](inputs)
                outputs = F.softmax(outputs, dim=1)
                p = outputs.gather(1, labels.view(-1, 1)).squeeze(1)
                complement = 1 - p
                p = torch.clamp(p, min=epsilon, max=1 - epsilon)
                complement = torch.clamp(complement, min=epsilon)
                score = torch.log(p) - torch.log(complement)
            score_list.append(score)
        score_list = torch.cat(score_list, dim=0)
        clients_score_list.append(score_list)

    clients_score_list = torch.stack(clients_score_list)
    data_len = clients_score_list[0].size(0)

    data_prob_list = []
    for data in range(data_len):
        data_score = clients_score_list[1:, data]
        mean = data_score.mean().item()
        std = data_score.std().item()
        if std == 0:
            std = epsilon
        normal_dist = Normal(mean, std)
        prob = normal_dist.cdf(clients_score_list[0, data]).item()
        data_prob_list.append(prob)

    return data_prob_list


def lira_mia(index_list, target, model_list, shadow_model_list, train_dataset, test_dataset):
    # Process the members and non-members
    train_loaders = LocalDataloaders(train_dataset, index_list, 64, ShuffleorNot=False)
    train_loader_length = len(train_loaders[target].dataset)
    test_loaders = sample_test_loader_by_label_distribution(train_loaders, test_dataset, target, batch_size=64)
    print(f"Train loader length: {train_loader_length}")
    print(f"Test loader length: {len(test_loaders.dataset)}")

    # distill LiRA mia
    if args.attack == "distill":
        in_prob_list = get_mia_score_distill(train_loaders[target], model_list, shadow_model_list, target)
        out_prob_list = get_mia_score_distill(test_loaders, model_list, shadow_model_list, target)

    if args.attack == "coop":
        in_prob_list = get_mia_score_coop(train_loaders[target], model_list, target)
        out_prob_list = get_mia_score_coop(test_loaders, model_list, target)

    return in_prob_list, out_prob_list


def distill_model(model, teacher_model, data_loader, device, epochs=20):
    model.train()
    teacher_model.eval()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = model(inputs)
            loss = criterion(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(data_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")
    return model


def random_sample_dataset(dataset, sample_size):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]
    return Subset(dataset, sampled_indices)


# Make sure the non-member and member share the same data distribution, so we can get a fair MIA results.
def sample_test_loader_by_label_distribution(train_loader, test_dataset, target, batch_size=64):
    label_counts = Counter()
    for _, labels in train_loader[target]:
        label_counts.update(labels.cpu().numpy())

    total_samples = sum(label_counts.values())
    label_distribution = {label: count / total_samples for label, count in label_counts.items()}

    test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    test_indices_by_label = {label: np.where(test_labels == label)[0] for label in label_distribution.keys()}

    sampled_test_indices = []
    for label, proportion in label_distribution.items():
        num_samples = int(proportion * len(train_loader[target].dataset))
        sampled_test_indices.extend(np.random.choice(test_indices_by_label[label], num_samples, replace=True))

    sampled_test_subset = Subset(test_dataset, sampled_test_indices)
    test_loader = DataLoader(sampled_test_subset, batch_size=batch_size, shuffle=False)

    return test_loader


epsilon = 1e-8
device = 'cuda'
args = args_parser()

# Load the models
model_list = []
for i in range(10):
    model = ResNet18(num_classes=10)
    model.load_state_dict(torch.load(f'../{args.alg}/{args.beta}/{args.dataset}/client_{i}_{args.epoch}.pt', map_location=device))
    model.to(device)
    model.eval()
    model_list.append(model)
for idx, model in enumerate(model_list):
    print(f'Model {idx} loaded and ready for inference.')

# Set the seed for data
rand_seed = 42
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)
random.seed(rand_seed)

# Prepare the data
data_dir = '../data/cifar10'
apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

# Partial the data
public_ratio = 0.2
len_train = len(train_dataset)
len_public = int(public_ratio * len_train)
len_train = len_train - len_public
train_dataset, public_dataset = torch.utils.data.random_split(train_dataset,[len_train, len_public])

# Select the member dict
index_dict = torch.load(f'./train_dict_{args.alg}_{args.dataset}_{args.beta}.pt')
index_list = index_dict

tpr_001_list=[]
tpr_01_list=[]
tpr_1_list=[]
accuracy_list = []
auc_list = []

# distill method
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
rng = np.random.default_rng(42)

tmp, public_dataset = torch.utils.data.random_split(public_dataset, [len(public_dataset)-5000, 5000])

for target in range(10):
    sample_size = int(0.8 * len(public_dataset))
    sampled_datasets = [random_sample_dataset(public_dataset, sample_size) for _ in range(args.num_shadows)]

    # distill the shadow models
    shadow_model_list = []
    teacher_model = model_list[target].to(device)
    for i, sampled_dataset in enumerate(sampled_datasets):
        model_save_path = f"../{args.alg}/{args.beta}/{args.dataset}/distill_{target}_{args.epoch}_{i}.pt"
        if os.path.exists(model_save_path):
            print(f"Model {i + 1} already exists, loading...")
            student_model = ResNet18(num_classes=10)
            student_model.load_state_dict(torch.load(model_save_path, map_location=device))
            student_model = student_model.to(device)
            shadow_model_list.append(student_model)
        else:
            print(f"Training model {i + 1} with sampled dataset...")
            data_loader = DataLoader(sampled_dataset, batch_size=64, shuffle=True)
            student_model = ResNet18(num_classes=10)
            student_model = student_model.to(device)
            distill_model(student_model, teacher_model, data_loader, device)
            torch.save(student_model.state_dict(), model_save_path)
            print(f"Model {i + 1} saved to {model_save_path}")
            shadow_model_list.append(student_model)

    # lira
    print("Attacking client:{}".format(target))
    in_prob_list, out_prob_list = lira_mia(index_list, target, model_list, shadow_model_list, train_dataset, test_dataset)
    in_prob_list = np.array(in_prob_list)
    out_prob_list = np.array(out_prob_list)
    print(np.mean(in_prob_list))
    print(np.mean(out_prob_list))
    labels = np.concatenate([np.ones(len(in_prob_list)), np.zeros(len(out_prob_list))])
    scores = np.concatenate([in_prob_list, out_prob_list])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    np.save(f'./result/{args.alg}_{args.dataset}_{args.beta}_{target}_fpr.npy', fpr)
    np.save(f'./result/{args.alg}_{args.dataset}_{args.beta}_{target}_tpr.npy', tpr)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")
    accuracies = []
    for threshold in thresholds:
        pred_labels = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, pred_labels)
        accuracies.append(accuracy)
    max_accuracy = max(accuracies)
    print(f"Max Accuracy: {max_accuracy}")

    tpr_001 = tpr[np.where(fpr <= 0.0001)[0][-1]]
    tpr_01 = tpr[np.where(fpr <= 0.001)[0][-1]]
    tpr_1 = tpr[np.where(fpr <= 0.01)[0][-1]]
    print("Target Client:{}, TPR @1%FPR: {}, TPR @0.1%FPR: {}, TPR @0.01%FPR: {}".format(target, tpr_1, tpr_01, tpr_001))
    accuracy_list.append(max_accuracy)
    auc_list.append(roc_auc)
    tpr_001_list.append(tpr_001)
    tpr_01_list.append(tpr_01)
    tpr_1_list.append(tpr_1)

mean_auc = np.mean(auc_list)
mean_accuracy = np.mean(accuracy_list)
mean_tpr_001 = np.mean(tpr_001_list)
mean_tpr_01 = np.mean(tpr_01_list)
mean_tpr_1 = np.mean(tpr_1_list)

print(f"accuracy:{mean_accuracy}")
print(f"auc:{mean_auc}")
print(f"Mean TPR @ 0.01% FPR: {mean_tpr_001}")
print(f"Mean TPR @ 0.1% FPR: {mean_tpr_01}")
print(f"Mean TPR @ 1% FPR: {mean_tpr_1}")







