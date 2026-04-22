import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001

# Try multiple lambda values
LAMBDA_VALUES = [1e-5, 1e-4, 1e-3]

THRESHOLD = 1e-2
