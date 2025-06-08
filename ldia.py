import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# kl loss
def calculate_kl_loss(result, ground_truth_distribution):
    result_kl = torch.log(result)
    ground_truth_distribution = torch.tensor(ground_truth_distribution)
    kl_loss = F.kl_div(result_kl, ground_truth_distribution, reduction='sum')
    return kl_loss


# tv loss
def total_variation_distance(p, q): # p is ground truth
    q = np.array(q)
    p = np.array(p)
    return 0.5 * np.sum(np.abs(p - q))


# tv loss
def Chebyshev(p, q): # p is ground truth
    q = np.array(q)
    p = np.array(p)
    return np.max(np.abs(p - q))


def mean_l1(p, q): # p is ground truth
    q = np.array(q)
    p = np.array(p)
    return np.mean(np.abs(p - q))


def plot_clients_distributions(final_results_list, ground_truth_list, kl_loss_list, tv_loss_list, chebyshev_list, mean_l1_list):
    selected_client = [3, 8, 0] # 0.1

    cols = 3
    rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    axes = axes.flatten()  # Flatten the axes array to easily index

    for num in range(3):
        client = selected_client[num]
        ax = axes[num]
        mean_values = torch.mean(final_results_list[client], dim=0)
        std_dev = torch.std(final_results_list[client], dim=0)

        # Convert to numpy for plotting
        mean_values = mean_values.numpy()
        std_dev = std_dev.numpy()
        ground_truth_distribution = ground_truth_list[client].numpy()

        x = range(len(mean_values))
        ax.plot(x, mean_values, color="steelblue", marker='*', label=r'$\hat{p}$(mean±std)')
        ax.fill_between(x, mean_values - std_dev, mean_values + std_dev, color="steelblue", alpha=0.2)
        ax.plot(x, ground_truth_distribution, color='red', marker='o', label='Ground Truth')
        ax.set_xticks(range(10))
        ax.set_title(f'KL-div: {kl_loss_list[client]:.2f}', fontsize=24)
        ax.set_xlabel('Label', fontsize=24)
        ax.set_ylabel('Percentage', fontsize=24)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='both', which='major', labelsize=18)  # 增大刻度字体
        ax.legend(fontsize=24, loc='upper right')

    plt.tight_layout()
    plt.savefig('./dsfl_result_alpha_0.1.png', dpi=300)

    KL = np.array(kl_loss_list)
    KL = np.mean(KL)
    print("kl", KL)

    TV = np.array(tv_loss_list)
    TV = np.mean(TV)
    print("tv", TV)

    chebyshev = np.array(chebyshev_list)
    mean_chebyshev = np.mean(chebyshev)
    print("chebyshev", mean_chebyshev)
    mean_l1 = np.array(mean_l1_list)
    mean_l1 = np.mean(mean_l1)
    print("mean_l1", mean_l1)


if __name__ == '__main__':
    data = torch.load('./FedMD_label_distribution_1.0.pt')   #For rebuttal
    label = torch.load('./FedMDground_truth_distribution_1.0.pt')  #For rebuttal

    final_results_list = []
    ground_truth_list = []
    kl_loss_list = []
    tv_loss_list = []
    chebshev_list = []
    mean_l1_list = []

    for client in range(10):
        ground_truth_distribution = label[client]
        ground_truth_list.append(ground_truth_distribution)

        final_result = []
        for epoch in range(9):
            epoch_data = data[epoch][client]
            inference_truth_distribution = torch.mean(epoch_data, dim=0)
            final_result.append(inference_truth_distribution)

        final_result = torch.stack(final_result)
        final_results_list.append(final_result)

        kl_loss = calculate_kl_loss(torch.mean(final_result, dim=0), ground_truth_distribution)
        tv_loss = total_variation_distance(ground_truth_distribution, torch.mean(final_result, dim=0))
        chebshev_loss = Chebyshev(ground_truth_distribution, torch.mean(final_result, dim=0))
        mean_l1_loss = mean_l1(ground_truth_distribution, torch.mean(final_result, dim=0))

        kl_loss_list.append(kl_loss)
        tv_loss_list.append(tv_loss)
        chebshev_list.append(chebshev_loss)
        mean_l1_list.append(mean_l1_loss)

    plot_clients_distributions(final_results_list, ground_truth_list, kl_loss_list, tv_loss_list, chebshev_list,mean_l1_list)




