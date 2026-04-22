# 🧠 Self-Pruning Neural Network Report

## 📖 Introduction

In this project, we implement a self-pruning neural network that dynamically removes unnecessary connections during training. Unlike traditional pruning techniques applied after training, this approach integrates pruning directly into the learning process.

---

## ⚙️ Methodology

A custom linear layer (`PrunableLinear`) is introduced, where each weight is associated with a learnable gate parameter. The effective weight is computed as:

effective_weight = weight × gate

The gates are obtained by applying a sigmoid function to learnable parameters.

---

## 📉 Sparsity Regularization

To encourage pruning, an L1-based sparsity loss is added:

Total Loss = Classification Loss + λ × Sparsity Loss

The sparsity loss is computed as the average of all gate values, which pushes them toward zero.

---

## 📊 Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.001  | 48.43       | 43.22       |
| 0.01   | 48.42       | 51.57       |
| 0.05   | 48.06       | 55.29       |

---

## 📈 Observations

- Increasing λ increases sparsity
- Higher sparsity slightly reduces accuracy
- The model successfully learns to remove redundant connections

---

## 📊 Visualization Insights

- Gate distribution plots show a strong concentration near zero
- Accuracy vs sparsity plot highlights the trade-off clearly

---

## 🧠 Conclusion

The self-pruning mechanism is effective in learning sparse representations. The model achieves a good balance between accuracy and efficiency without requiring post-training pruning.

---

## 🚀 Future Work

- Extend pruning to convolutional layers
- Explore structured pruning (neuron/channel level)
- Use advanced gating techniques like Gumbel Softmax
