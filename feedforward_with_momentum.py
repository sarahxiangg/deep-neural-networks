import numpy as np


class FeedforwardLayer():

    def __init__(
        self,
        m: int,
        n: int,
        lr: float,
        initial_weights: np.ndarray = None,
        initial_biases: np.ndarray = None,
        momentum: float = 0.9
    ):

        self.m = m
        self.n = n

        self.lr = lr
        self.momentum = momentum

        if initial_weights is not None:
            self.weights = initial_weights
        else:
            self.weights = np.random.uniform(-1, 1, (n, m))

        if initial_biases is not None:
            self.biases = initial_biases
        else:
            self.biases = np.random.uniform(-1, 1, n)

        self.weight_velocity = np.zeros((n, m))
        self.bias_velocity = np.zeros(n)

        self.last_input = None

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_biases(self) -> np.ndarray:
        return self.biases

    def get_lr(self) -> float:
        return self.lr

    def forward(self, x) -> np.ndarray:

        self.last_input = x

        output = self.weights @ x
        output = output + self.biases

        return output

    def backward(self, grads: np.ndarray) -> np.ndarray:

        grad_weights = np.outer(grads, self.last_input)

        grad_biases = grads

        grad_input = self.weights.T @ grads

        weight_update = self.lr * grad_weights
        bias_update = self.lr * grad_biases

        self.weight_velocity = (
            self.momentum * self.weight_velocity
        ) - weight_update

        self.bias_velocity = (
            self.momentum * self.bias_velocity
        ) - bias_update

        self.weights = self.weights + self.weight_velocity
        self.biases = self.biases + self.bias_velocity

        return grad_input
