import numpy as np

class SoftmaxLayer():
    def __init__(self, n):
        self.n = n
        self.m = n
        self.res = None

    def forward(self, x):
        shifted = x - np.max(x)
        exp_values = np.exp(shifted)
        self.res = exp_values / np.sum(exp_values)
        return self.res

    def backward(self, grads: np.ndarray) -> np.ndarray:
        return grads