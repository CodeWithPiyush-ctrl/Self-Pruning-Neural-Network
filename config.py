import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 10
LR = 0.0003

# tuned lambdas
LAMBDA_VALUES = [1e-3, 1e-2, 5e-2]

THRESHOLD = 0.1
