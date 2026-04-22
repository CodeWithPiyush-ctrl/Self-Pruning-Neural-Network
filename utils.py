import torch
import matplotlib.pyplot as plt

def sparsity_loss(model):
    loss = 0
    for gates in model.get_all_gates():
        loss += torch.sum(gates)
    return loss


def calculate_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for gates in model.get_all_gates():
        total += gates.numel()
        pruned += torch.sum(gates < threshold).item()

    return 100 * pruned / total


def plot_gate_distribution(model, path):
    all_gates = []
    for gates in model.get_all_gates():
        all_gates.extend(gates.detach().cpu().numpy().flatten())

    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.savefig(path)
    plt.close()
