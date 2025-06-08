# Unveiling Client Privacy Leakage from Public Dataset Usage in Federated Distillation
The source code of the paper **"Unveiling Client Privacy Leakage from Public Dataset Usage in Federated Distillation"**, in Proceedings on Privacy Enhancing Technologies(**PoPETs 2025**).

# Usage
Experimental procedure:
- STEP 1: Perform the Federated Distillation
- STEP 2: Perform the Label Distribution Attack
- STEP 3: Perform the Co-op LiRA/Distillation-based LiRA.
### Configure the environment needed for the experiment
```
pip install -r requirements.txt
```
- Create folders to store the MIA results
```
cd Privacy_PDAFD
mkdir result
```
### Example(FedMD, CIFAR10(Private/Public), alpha = 1)
- **STEP 1**: Perform the Federated Distillation
```
python main.py --dataset 'CIFAR10' --batch_size 64  --digest_batch_size 128 --num_rounds 10 --lr 0.001 --co_lr 0.001 --num_clients 10 --num_classes 10 --sampling_rate 1 --pre_ep 20 --digest_ep 10 --revisit_ep 3 --beta 1 --seed 42 --alg 'FedMD' --model 'resnet18'  --public_part 0.2
```
- **STEP 2**: Perform the Label Distribution Attack
```
python ldia.py
```
- **STEP 3(Co-op LiRA)**: Perform the Co-op LiRA.
```
python mia.py --alg FedMD --beta 1 --dataset CIFAR10 --epoch 0 --num_shadows 16 --attack coop
```
- **STEP 3(Distillation-based LiRA)**: Perform Distillation-based LiRA.
```
python mia.py --alg FedMD --beta 1 --dataset CIFAR10 --epoch 0 --num_shadows 16 --attack distill
```

### Details
- **STEP 1**: Train the target model
```
python main.py --dataset [dataset] --batch_size [batch_size] --digest_batch_size [batch_size] --num_rounds [rounds] --lr [learning_rate] --co_lr [learning_rate]
               --num_clients [clients] --num_classes [classes] --sampling_rate [sampling rate] --pre_ep [epochs] --digest_ep [epochs] --revisit_ep [epochs] 
               --beta [alpha_value] --seed [random seed] --alg [FD_alg] --model [model_architecture]  --public_part [public_dataset_ratio]


optional arguments:
    --dataset              //experiment dataset
    --batch_size           //batch size for local updates phase
    --digest_batch_size    //batch size for knowledge distillation phase
    --num_rounds           //number of collaboration training rounds
    --lr                   //learning rate for local updates phase
    --co_lr                //learning rate for knowledge distillation phase
    --num_clients          //number of clients
    --num_classes          //number of classes for training dataset
    --sampling_rate        //the ratio of activated clients
    --pre_ep               //training epochs for local updates phase(round 0)
    --digest_ep            //training epochs for knowledge distillation phase
    --revisit_ep           //training epochs for local updates phase(after round 0)
    --beta                 //alpha value for Dirichlet distribution
    --seed                 //random seed
    --alg                  //Federated Distillation Algorithm
    --model                //Client's model architecture
    --public_part          //public dataset size(dataset size*public_part)
```

- **STEP 2**: Perform the Label Distribution Attack
```
python ldia.py
```

- **STEP 3**: Perform the Co-op LiRA/Distillation-based LiRA.
```
python mia.py --alg [FD alg] --beta [alpha value] --dataset [dataset] 
              --epoch [target collaborated training round] --num_shadows [number of shadow models] --attck [attack method]

optional arguments:
  --alg                  //Federated Distillation Algorithm
  --beta                 //alpha value for Dirichlet distribution
  --epoch                //the attack epoch(collaborated training round)
  --num_shadows          //number of shadow models
  --attack               //attack methods(coop, distill)
```