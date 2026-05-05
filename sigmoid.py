import numpy as np

class SigmoidLayer():
    def __init__(self, n):
        self.n = n
        self.m = n
        self.res = None

    def forward(self, x):
        self.res = 1 / (1 + np.exp(-x))
        return self.res

    def backward(self, grads: np.ndarray) -> np.ndarray:

        sig= self.res

        der = sig * (1 - sig)

        grad_input = grads * der

        return grad_input
