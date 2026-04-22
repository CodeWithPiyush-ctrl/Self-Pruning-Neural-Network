=
         SELF-PRUNING NEURAL NETWORK
         Dynamic Connection Pruning via Learnable Gates & L1 Regularization
================================================================================

REPOSITORY   : CodeWithPiyush-ctrl/Self-Pruning-Neural-Network
LANGUAGE     : Python 3
FRAMEWORK    : PyTorch
DATASET      : CIFAR-10 (auto-downloaded)

--------------------------------------------------------------------------------
OVERVIEW
--------------------------------------------------------------------------------

This project implements a Self-Pruning Neural Network trained on CIFAR-10.
Instead of the traditional "train -> prune -> fine-tune" pipeline, the model
dynamically learns WHICH connections to keep or discard entirely during the
forward/backward pass.

Each weight matrix has an associated set of learnable gate_scores. These scores
are passed through a sigmoid function to produce gate values in [0, 1]. Gates
are multiplied element-wise onto the weights before each linear operation.
An L1-style sparsity loss penalizes large gate values, pushing unnecessary
connections toward zero over time.

The model trains three different sparsity levels by sweeping lambda (the
regularization coefficient), and reports Accuracy vs. Sparsity tradeoffs.

--------------------------------------------------------------------------------
HOW IT WORKS
--------------------------------------------------------------------------------

1. LEARNABLE GATES  (model.py - PrunableLinear)
   - Each PrunableLinear layer holds a weight matrix AND a gate_scores matrix
     of the same shape (out_features x in_features).
   - During forward pass:
         gates = sigmoid(gate_scores.clamp(-10, 10) / 0.1)
         pruned_weights = weight * gates
         output = F.linear(x, pruned_weights, bias)
   - The temperature 0.1 in the sigmoid produces near-binary gate values,
     pushing each gate toward either ~0 (pruned) or ~1 (kept).

2. SPARSITY LOSS  (utils.py - sparsity_loss)
   - Computes the mean of all gate values across all layers.
   - Scaled by a factor of 5:
         sparsity_loss = 5 * mean(all_gates)
   - Added to the classification loss during training.

3. ADAPTIVE LAMBDA  (train.py)
   - Sparsity regularization is applied gradually:
         adaptive_lambda = lambda_val * (epoch / total_epochs)
   - This ramps from 0 to lambda_val, letting the model learn good
     representations before compression is aggressively enforced.

4. TRAINING LOOP SUMMARY
   - Dataset      : CIFAR-10 (50k train / 10k test)
   - Optimizer    : Adam (lr = 0.0003)
   - Loss         : CrossEntropyLoss + adaptive_lambda * sparsity_loss
   - Grad Clip    : max norm = 1.0 (prevents gradient explosion)
   - Epochs       : 10 per lambda run
   - Batch Size   : 128

5. SPARSITY CALCULATION  (utils.py - calculate_sparsity)
   - After training, gates below THRESHOLD = 0.1 are counted as pruned.
   - Sparsity (%) = 100 * pruned_connections / total_connections

--------------------------------------------------------------------------------
MODEL ARCHITECTURE  (model.py - PrunableNN)
--------------------------------------------------------------------------------

  Input: CIFAR-10 image -> flattened to 3072 (32x32x3)

  PrunableLinear(3072 -> 512)   + ReLU
  PrunableLinear(512  -> 256)   + ReLU
  PrunableLinear(256  -> 10)    (logits, no activation)

  Total gate parameters:
    Layer 1: 3072 x 512  = 1,572,864 gates
    Layer 2: 512  x 256  =   131,072 gates
    Layer 3: 256  x 10   =     2,560 gates
    -----------------------------------
    Total  :               1,706,496 gates (mirrors total weight count)

--------------------------------------------------------------------------------
PROJECT STRUCTURE
--------------------------------------------------------------------------------

  Self-Pruning-Neural-Network-main/
  |-- model.py            # PrunableLinear + PrunableNN architecture
  |-- train.py            # Full training loop, lambda sweep, result plots
  |-- utils.py            # sparsity_loss, calculate_sparsity, gate plot
  |-- config.py           # Hyperparameters (device, epochs, lr, lambdas)
  |-- requirements.txt    # torch, torchvision, matplotlib
  |-- plots/
      |-- Metrics.txt     # Recorded accuracy & sparsity results
      |-- *.png           # Gate distribution and accuracy/sparsity plots

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------

  Prerequisites:
    Python 3.8+

  Install dependencies:
    pip install -r requirements.txt

  Contents of requirements.txt:
    torch
    torchvision
    matplotlib

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------

  Run the full training + evaluation sweep:

    python train.py

  This will:
    1. Auto-download CIFAR-10 into ./data/
    2. Train three separate models for each lambda in [0.001, 0.01, 0.05]
    3. Save gate distribution plots to  results/plots/
    4. Save accuracy vs sparsity plot to results/plots/accuracy_vs_sparsity.png
    5. Save metrics summary to          results/metrics.txt

  To change hyperparameters, edit config.py:

    BATCH_SIZE    = 128
    EPOCHS        = 10
    LR            = 0.0003
    LAMBDA_VALUES = [1e-3, 1e-2, 5e-2]   # sparsity regularization strengths
    THRESHOLD     = 0.1                   # gate cutoff for pruning count

--------------------------------------------------------------------------------
RESULTS  (from plots/Metrics.txt)
--------------------------------------------------------------------------------

  Lambda       Accuracy (%)    Sparsity (%)
  -----------  --------------  ---------------
  0.001        48.43           43.22
  0.010        48.42           51.57
  0.050        48.06           55.29

  Key observations:
  - Increasing lambda drives higher sparsity (43% -> 55% pruned connections)
    with only ~0.37% accuracy degradation across the entire range.
  - The model prunes 55% of all connections while still achieving ~48%
    accuracy on CIFAR-10 in just 10 training epochs.
  - Note: ~48% accuracy is typical for a small feedforward (non-convolutional)
    network on CIFAR-10. Adding CNN layers would significantly improve accuracy.

--------------------------------------------------------------------------------
CONFIGURATION REFERENCE  (config.py)
--------------------------------------------------------------------------------

  Parameter         Default           Description
  ----------------  ----------------  -----------------------------------------
  DEVICE            auto              "cuda" if GPU available, else "cpu"
  BATCH_SIZE        128               Mini-batch size for DataLoader
  EPOCHS            10                Training epochs per lambda run
  LR                0.0003            Adam optimizer learning rate
  LAMBDA_VALUES     [1e-3,1e-2,5e-2]  Sparsity regularization coefficients
  THRESHOLD         0.1               Gates below this are counted as pruned

--------------------------------------------------------------------------------
KEY DESIGN DECISIONS
--------------------------------------------------------------------------------

  Temperature scaling (/ 0.1 in sigmoid):
    Sharpens gate values toward 0 or 1, making pruning decisions more decisive
    rather than keeping many soft mid-range values.

  Gradient clipping (max_norm=1.0):
    Prevents exploding gradients when gate_scores reach extreme values during
    the steep regions of the scaled sigmoid.

  Adaptive lambda ramping:
    Avoids aggressive early pruning that could damage representations before
    the model has had time to learn meaningful features.

  Gate score clamping (clamp(-10, 10)):
    Prevents numerical overflow in the sigmoid when scores drift far from zero.

--------------------------------------------------------------------------------
LIMITATIONS & POSSIBLE IMPROVEMENTS
--------------------------------------------------------------------------------

  - Feedforward only: No convolutional layers. Adding a CNN backbone would
    dramatically improve CIFAR-10 accuracy.

  - Unstructured pruning: Pruning individual weights doesn't reduce inference
    latency without sparse tensor hardware/library support. Structured pruning
    (entire neurons or channels) would give real wall-clock speedups.

  - No model checkpointing: Trained models are not saved. Adding torch.save()
    would allow inspecting or deploying pruned models after training.

  - Short training: 10 epochs with a fixed learning rate. A cosine LR schedule
    and more epochs would likely improve both accuracy and sparsity quality.

  - Lambda sweep is manual: A Bayesian or grid search over lambda could find a
    better point on the accuracy-sparsity Pareto frontier.

--------------------------------------------------------------------------------
REFERENCES
--------------------------------------------------------------------------------

  [1] Han et al., "Learning both Weights and Connections for Efficient
      Neural Networks", NeurIPS 2015.
  [2] Louizos et al., "Learning Sparse Neural Networks through L0
      Regularization", ICLR 2018.
  [3] LeCun et al., "Optimal Brain Damage", NeurIPS 1990.

--------------------------------------------------------------------------------
LICENSE
--------------------------------------------------------------------------------

  MIT License - free to use, modify, and distribute with attribution.

--------------------------------------------------------------------------------
CONTACT
--------------------------------------------------------------------------------

  GitHub : https://github.com/CodeWithPiyush-ctrl/Self-Pruning-Neural-Network

================================================================================
