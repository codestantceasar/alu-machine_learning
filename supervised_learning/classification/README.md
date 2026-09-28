hhhhhhhhhh# Classification

This project introduces the fundamentals of neural networks and deep learning using NumPy. The goal is to build binary and multiclass classification models from scratch while understanding forward propagation, cost calculation, gradient descent, training, and model persistence.

## Learning Objectives

By completing this project, you should be able to:

* Explain what supervised learning is
* Understand binary and multiclass classification
* Build neurons and neural networks from scratch
* Implement forward propagation
* Calculate logistic regression cost
* Perform gradient descent
* Train neural networks
* Use one-hot encoding and decoding
* Save and load trained models
* Understand activation functions such as sigmoid and tanh

## Requirements

* Ubuntu 20.04 LTS
* Python 3.9+
* NumPy
* Matplotlib
* Pycodestyle

## Files

| File                        | Description                            |
| --------------------------- | -------------------------------------- |
| `0-neuron.py`               | Defines a single neuron                |
| `1-neuron.py`               | Adds private attributes and getters    |
| `2-neuron.py`               | Forward propagation                    |
| `3-neuron.py`               | Cost calculation                       |
| `4-neuron.py`               | Evaluation                             |
| `5-neuron.py`               | Gradient descent                       |
| `6-neuron.py`               | Neuron training                        |
| `7-neuron.py`               | Verbose training                       |
| `8-neural_network.py`       | Neural network with one hidden layer   |
| `9-neural_network.py`       | Forward propagation for neural network |
| `10-neural_network.py`      | Cost calculation                       |
| `11-neural_network.py`      | Evaluation                             |
| `12-neural_network.py`      | Gradient descent                       |
| `13-neural_network.py`      | Training                               |
| `14-neural_network.py`      | Verbose and graphical training         |
| `15-neural_network.py`      | Saving and loading models              |
| `16-deep_neural_network.py` | Deep neural network initialization     |
| `17-deep_neural_network.py` | Private attributes and getters         |
| `18-deep_neural_network.py` | Forward propagation                    |
| `19-deep_neural_network.py` | Cost calculation                       |
| `20-deep_neural_network.py` | Evaluation                             |
| `21-deep_neural_network.py` | Gradient descent                       |
| `22-deep_neural_network.py` | Training                               |
| `23-deep_neural_network.py` | Verbose and graphical training         |
| `24-one_hot_encode.py`      | One-hot encoding                       |
| `25-one_hot_decode.py`      | One-hot decoding                       |
| `26-deep_neural_network.py` | Model persistence                      |
| `27-deep_neural_network.py` | Multiclass classification              |
| `28-deep_neural_network.py` | Multiple activation functions          |

## Concepts Covered

### Forward Propagation

Forward propagation computes the output of each layer by applying weights, biases, and activation functions.

### Cost Function

The logistic regression cost function measures prediction error:

$$
J = -\frac{1}{m}\sum(Y \log(A) + (1-Y)\log(1-A))
$$

### Gradient Descent

Gradient descent updates weights and biases to minimize the cost function.

### Activation Functions

#### Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Used for binary classification outputs.

#### Tanh

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Provides outputs between -1 and 1 and often improves learning speed.

### One-Hot Encoding

Converts class labels into vectors where only one element is set to 1 and all others are 0.

Example:

```python
Labels:
[2, 0, 1]

One-hot:
[[0, 1, 0],
 [0, 0, 1],
 [1, 0, 0]]
```

## Running the Files

Example:

```bash
python3 23-main.py
```

Check style compliance:

```bash
pycodestyle *.py
```

## Author

Constantine Akas

African Leadership University (ALU)

Software Engineering
