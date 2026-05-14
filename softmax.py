import numpy as np


class SoftmaxLayer():
    def __init__(self, n):
        # initialise softmax layer size and store outputs
        self.n = n
        self.m = n
        self.res = None

    def forward(self, x):
        # convert outputs into probability distribution
        # subtract max value for numerical stability
        shifted = x - np.max(x)
        exp_values = np.exp(shifted)
        self.res = exp_values / np.sum(exp_values)

        return self.res

    def backward(self, grads: np.ndarray) -> np.ndarray:
        # return gradients during backpropagation
        # gradient simplifies with softmax + cross entropy
        return grads