import torch
import matplotlib.pyplot as plt
import config

def sparsity_loss(model):
    """
    Computes L1-based sparsity loss over all gates.
    Encourages gates to move toward zero → pruning.
    """
    loss = 0
    count = 0
    for gates in model.get_all_gates():
        loss += torch.sum(gates)
        count += gates.numel()
    return 5 * (loss / count)


def calculate_sparsity(model):
    """
    Calculates percentage of gates below threshold (pruned weights).
    """
    total = 0
    pruned = 0

    for gates in model.get_all_gates():
        total += gates.numel()
        pruned += torch.sum(gates < config.THRESHOLD).item()

    return 100 * pruned / total


def plot_gate_distribution(model, path):
    all_gates = []
    for gates in model.get_all_gates():
        all_gates.extend(gates.detach().cpu().numpy().flatten())

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title("Gate Distribution")
    plt.savefig(path)
    plt.close()
