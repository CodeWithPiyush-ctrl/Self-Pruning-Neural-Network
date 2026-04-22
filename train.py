import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

from model import PrunableNN
from utils import sparsity_loss, calculate_sparsity, plot_gate_distribution
import config

os.makedirs("results/plots", exist_ok=True)

def train(lambda_val):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.BATCH_SIZE, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config.BATCH_SIZE)

    model = PrunableNN().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            outputs = model(images)

            cls_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            adaptive_lambda = lambda_val * (epoch / config.EPOCHS)

            loss = cls_loss + adaptive_lambda * sp_loss

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(trainloader.dataset)
        print(f"[Lambda {lambda_val}] Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    sparsity = calculate_sparsity(model)

    plot_gate_distribution(model, f"results/plots/gate_lambda_{lambda_val}.png")

    return accuracy, sparsity


if __name__ == "__main__":

    results = []

    for lam in config.LAMBDA_VALUES:
        acc, sp = train(lam)

        print(f"Lambda: {lam}, Accuracy: {acc:.2f}, Sparsity: {sp:.2f}")
        results.append((lam, acc, sp))

    # Save metrics
    with open("results/metrics.txt", "w") as f:
        f.write("Lambda\tAccuracy\tSparsity\n")
        for r in results:
            f.write(f"{r[0]}\t{r[1]:.2f}\t{r[2]:.2f}\n")

    # Plot Accuracy vs Sparsity
    lambdas = [r[0] for r in results]
    accuracies = [r[1] for r in results]
    sparsities = [r[2] for r in results]

    plt.figure()
    plt.plot(sparsities, accuracies, marker='o')
    plt.xlabel("Sparsity (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Sparsity Tradeoff")
    plt.savefig("results/plots/accuracy_vs_sparsity.png")
    plt.show()
