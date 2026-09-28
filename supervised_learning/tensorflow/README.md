# TensorFlow

This project introduces the fundamentals of building neural networks using TensorFlow 1.12. It covers creating placeholders, building layers, forward propagation, calculating accuracy and loss, training neural networks, saving models, and evaluating trained models.

## Learning Objectives

By completing this project, you should understand:

* What TensorFlow is
* What graphs and sessions are
* What tensors, variables, constants, and placeholders are
* How to create and use TensorFlow operations
* How to build a neural network in TensorFlow
* How to calculate accuracy and loss
* How to train a model using Gradient Descent
* How to save and restore trained models
* How to use graph collections

## Requirements

* Ubuntu 16.04 LTS
* Python 3.5
* TensorFlow 1.12
* NumPy 1.15
* pycodestyle 2.4

## Files

| File                       | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `0-create_placeholders.py` | Creates placeholders for input data and labels |
| `1-create_layer.py`        | Creates a neural network layer                 |
| `2-forward_prop.py`        | Builds the forward propagation graph           |
| `3-calculate_accuracy.py`  | Calculates prediction accuracy                 |
| `4-calculate_loss.py`      | Calculates softmax cross-entropy loss          |
| `5-create_train_op.py`     | Creates the training operation                 |
| `6-train.py`               | Builds, trains, and saves a neural network     |
| `7-evaluate.py`            | Loads and evaluates a saved model              |

## Usage

Run the provided test files:

```bash
python3 0-main.py
python3 1-main.py
python3 2-main.py
python3 3-main.py
python3 4-main.py
python3 5-main.py
python3 6-main.py
python3 7-main.py
```

## Author

Constantine Akas
