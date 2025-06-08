import torch
import numpy as np
import os,sys,os.path
from tensorboardX import SummaryWriter
import pickle
import hashlib


from models import ResNet18,ShuffLeNet, Smallnet, CNN, MLP
from sampling import LocalDataset, LocalDataloaders, partition_data, generate_shadow_training_data
from option import args_parser
from Server.ServerFedMD import ServerFedMD


print(torch.__version__)
torch.cuda.is_available()
np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)

args = args_parser()
print(args)
args_hash = ''
for k,v in vars(args).items():
    if k == 'eval_only':
        continue
    args_hash += str(k)+str(v)

args_hash = hashlib.sha256(args_hash.encode()).hexdigest()

attack_model_train_data_path = './attack_model_train.pkl'
attack_model_test_data_path = './attack_model_test.pkl'
attack_model_test_data_label_path = './attack_model_test_label.pkl'

if not os.path.exists(attack_model_train_data_path):
    train_dataset, testset, public_dataset, dict_users, dict_users_test = partition_data(n_users=args.num_clients, alpha=args.beta, rand_seed=args.seed,dataset=str(args.dataset),public_ratio=args.public_part)
    torch.save(dict_users, f'./train_dict_{args.alg}_{args.dataset}_{args.beta}.pt')
    train_loaders = LocalDataloaders(train_dataset, dict_users, args.batch_size, ShuffleorNot=True)
    test_loader = LocalDataloaders(testset, dict_users_test, args.batch_size, ShuffleorNot=False)

    # Print the distribution of Clients
    test_label = []
    for idx in range(args.num_clients):
        counts = [0] * args.num_classes
        for batch_idx, (X, y) in enumerate(train_loaders[idx]):
            batch = len(y)
            y = np.array(y)
            for i in range(batch):
                counts[int(y[i])-args.num_classes] += 1
        total = 0
        for count in counts:
            total = total + count
        for i in range(args.num_classes):
            counts[i] = counts[i] / total
        formatted_list = [f"{x:.2f}" for x in counts]
        print('Client {} data distribution(total {}):'.format(idx, total))
        print(formatted_list)
        counts = torch.tensor(counts)
        test_label.append(counts)
    torch.stack(test_label)

    logger = SummaryWriter('./logs')
    checkpoint_dir = './checkpoint/'+ args.dataset + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir+'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)
    print('Checkpoint dir:', checkpoint_dir)

    print(args.model)
    if args.model == 'CNN':
        global_model = CNN(num_classes=args.num_classes)
    if args.model == 'resnet18':
        print('Using the model resnet18...')
        global_model = ResNet18(num_classes=args.num_classes)
    if args.model == 'shufflenet':
        global_model = ShuffLeNet(args, code_length=args.code_len, num_classes = args.num_classes)
    if args.model == 'smallnet':
        print('Using the model smallnet...')
        global_model = Smallnet(num_classes=args.num_classes)
    if args.model == 'MLP':
        global_model = MLP(num_classes=args.num_classes)

    print('# model parameters:', sum(param.numel() for param in global_model.parameters()))
    global_model.to(device)

    if args.alg == 'FedMD':
        server = ServerFedMD(args, global_model, train_loaders, test_loader, public_dataset, logger, device)

    server.Create_Clints()
    # attack_train_data, attack_test_data = server.train()
    attack_test_data, attack_test_data_label = server.train()

    torch.save(attack_test_data, f'{args.alg}_label_distribution_{args.beta}.pt')
    torch.save(test_label, f'{args.alg}ground_truth_distribution_{args.beta}.pt')


