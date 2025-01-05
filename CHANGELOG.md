# Changelog

## Unreleased

### Added

- create `io` module for load/save model weights, training history, datasets, and evaluation results
- create `plot` module for data visualization
- add more model definition including:
    - AlexNet
    - VGG
    - ResNet
- add example for CIFAR-10 classification task

## v0.1.0 - 2024-12-23

### Initial Release

- create `helper` module for model training/evaluation and data loading
- create `models` module for model definition which includes:
    - `perceptron.py`: multi-layer perceptron (MLP)
    - `lenet5.py`: LeNet-5 CNN
    - `quantized.py`: interface to define quantized models
- create an example for training an MLP on MNIST dataset
