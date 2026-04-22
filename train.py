import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import PrunableNN
from utils import sparsity_loss, calculate_sparsity, plot_gate_distribution
import config

def train(lambda_val):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.BATCH_SIZE)

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

            # adaptive lambda (unique idea 🔥)
            adaptive_lambda = lambda_val * (epoch / config.EPOCHS)

            loss = cls_loss + adaptive_lambda * sp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

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

    plot_gate_distribution(model, f"results/plots/lambda_{lambda_val}.png")

    return accuracy, sparsity


if __name__ == "__main__":
    for lam in config.LAMBDA_VALUES:
        acc, sp = train(lam)

        print(f"Lambda: {lam}, Accuracy: {acc:.2f}, Sparsity: {sp:.2f}")
