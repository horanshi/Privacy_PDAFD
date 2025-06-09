# Artifact Appendix

Paper title: **Unveiling Client Privacy Leakage from Public Dataset Usage in Federated Distillation**

Artifacts HotCRP Id: **#17**

Requested Badge: **Available**, **Functional**, **Reproduced**

## Description
The source code of the paper "Unveiling Client Privacy Leakage from Public Dataset Usage in Federated Distillation", in Proceedings on Privacy Enhancing Technologies(PoPETs 2025).

In this codebase, we provide code for conducting Federated Distillation training (e.g., FedMD), as well as code for performing Label Distribution Attacks and Membership Inference Attacks (Co-op LiRA and Distillation-based LiRA) against clients participating in FD training when the server acts as an attacker.

### Security/Privacy Issues and Ethical Concerns
There are no Security/Privacy Issues and Ethical Concerns.

## Basic Requirements
### Hardware Requirements
No specific hardware is required. experiments are conducted on general-purpose machines equipped with Intel Xeon Silver 4208 CPU@2.10 GHz, Quadro RTX 5000 GPU, and 16 GB RAM.

### Software Requirements
All packages are included in ```requirements.txt```. 

### Estimated Time and Storage Consumption
- Estimated Time consumption:
Federated distillation training: 2 hours;
Label distribution attacks: 2 minutes;
Co-op LiRA: 10 minutes/client;
Distillation-basd LiRA: 40 minutes/client.

- Estimated storage consumption: 30GB.



## Environment
### Accessibility
GitHub Repository: https://github.com/horanshi/Privacy_PDAFD

### Set up the environment
- Configure the environment needed for the experiment
```
pip install -r requirements.txt
```
- Create folders to store the MIA results
```
cd Privacy_PDAFD
mkdir result
```

### Testing the Environment 
List all installed packages and compare with ```requirements.txt```.
```
pip list
```

## Artifact Evaluation
### Main Results and Claims
#### Main Result 1: Label distribution leakage

When clients upload logits following inference on public datasets, these values may inadvertently reveal the label distribution characteristics of their private datasets.
Please refer to ```Experiment 2``` to see how to reproduce the result.

#### Main Result 2: Membership information leakage

Using the logits uploaded by clients from inference on public datasets, the server can also leverage these logit values to conduct precise membership inference attacks against clients.
Please refer to ```Experiment 3 and 4``` to see how to reproduce the result.

### Experiments 
#### Experiment 1: Perform the Federated Distillation
Select one Federated Distillation training algorithm(eg. FedMD), decide the number of clients, and perform the federated distillation training on the clients. 
```
python main.py --dataset 'CIFAR10' --batch_size 64  --digest_batch_size 128 --num_rounds 10 --lr 0.001 --co_lr 0.001 --num_clients 10 --num_classes 10 --sampling_rate 1 --pre_ep 20 --digest_ep 10 --revisit_ep 3 --beta 1 --seed 42 --alg 'FedMD' --model 'resnet18'  --public_part 0.2
```
For the optional arguments, please refer to the ```README.md``` document.

#### Experiment 2: Perform the Label Distribution Attack
Perform the Label Distribution Attack on all the clients.
```
python ldia.py
```
For the optional arguments, please refer to the ```README.md``` document.

Expected results: We can observe that the predicted label distribution of the target client's private dataset is very close to the ground truth label distribution, with a small KL divergence value between them.
For specific results, please refer to the Table 3 and Figure 7 in the paper.


#### Experiment 3: Perform the Co-op LiRA.
Perform the Co-op LiRA for all the clients.
```
python mia.py --alg FedMD --beta 1 --dataset CIFAR10 --epoch 0 --num_shadows 16 --attack coop
```
For the optional arguments, please refer to the ```README.md``` document.

Expected results: We can observe that when using other clients' private models as reference models, the server can perform precise membership inference attacks against target clients, with very high True Positive Rate in the low False Positive Rate region.
For specific results, please to the Table 4 in the paper.

#### Experiment 3: Perform Distillation-based LiRA.
Decide the number of shadow models and perform the Distillation-based LiRA for all the clients.
```
python mia.py --alg FedMD --beta 1 --dataset CIFAR10 --epoch 0 --num_shadows 16 --attack distill
```
For the optional arguments, please refer to the ```README.md``` document.

Expected results: We can observe that when distilling target clients' models using public datasets to obtain reference models, the server can perform precise membership inference attacks against target clients using these distilled reference models, with very high True Positive Rate in the low False Positive Rate region. 
For specific results, please refer to the Table 5 and Figure 9 in the paper.


## Limitations
Due to time constraints in organizing our codebase, we have not yet elegantly integrated the DSFL and Cronus FD frameworks into this repository to allow reviewers to easily switch between DSFL and Cronus frameworks for evaluation purposes. 
So we used the FedMD framework as an example for illustration.
But due to the algorithmic similarities between FedMD and both DSFL and Cronus, as well as their respective use of public datasets, they will exhibit similar privacy leakage characteristics in the final privacy evaluation.
We plan to integrate the DSFL and Cronus FD frameworks into this GitHub repository in future updates.

## Notes on Reusability
Our framework can be utilized as an evaluation tool to assess the degree of privacy leakage in other federated distillation or federated learning frameworks.