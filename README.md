# NumPy Neural Network

A clean, educational implementation of a two-layer Neural Network built from scratch using **NumPy**, supporting both **regression** and **binary classification** tasks, alongside benchmark comparisons against `scikit-learn` models.

## Features

- **Custom `NeuralNetwork` Class**: Implemented purely with `numpy` (no deep learning frameworks).
- **Activation Functions & Derivatives**:
  - `relu` (hidden layer)
  - `linear` (regression output)
  - `sigmoid` (binary classification output)
- **Loss Functions & Derivatives**:
  - Mean Squared Error (`mse`)
  - Binary Cross-Entropy (`cross_entropy`)
- **Full Training Loop**: Forward propagation, backpropagation with gradient descent, periodic loss logging, and prediction methods.
- **Scikit-Learn Benchmarks**: Compares custom network performance against `LinearRegression`, `MLPRegressor`, `LogisticRegression`, and `MLPClassifier`.

## Architecture

| Component | Regression Setup | Classification Setup |
| :--- | :--- | :--- |
| **Input Layer** | 3 features | 10 features |
| **Hidden Layer** | 128 neurons (ReLU) | 128 neurons (ReLU) |
| **Output Layer** | 1 neuron (Linear) | 1 neuron (Sigmoid) |
| **Loss Function** | Mean Squared Error (MSE) | Binary Cross-Entropy |
| **Learning Rate** | `0.001` | `0.001` |
| **Epochs** | `5000` | `5000` |

## Results

### Regression (`make_regression`, 1,000 samples)
| Model | R² Score |
| :--- | :--- |
| Custom `NeuralNetwork` | `0.975` |
| `LinearRegression` | `0.976` |
| `MLPRegressor` | `0.976` |

### Binary Classification (`make_classification`, 1,000 samples)
| Model | Accuracy |
| :--- | :--- |
| Custom `NeuralNetwork` | `0.908` |
| `LogisticRegression` | `0.792` |
| `MLPClassifier` | `0.940` |

## Getting Started

### Prerequisites

Install the required Python libraries:

```bash
pip install numpy scikit-learn
