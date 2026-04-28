import numpy as np

class FeedforwardLayer():
    def __init__(self, m: int, n: int, lr: float, 
                 initial_weights: np.ndarray = None, 
                 initial_biases: np.ndarray = None):
        
        self.m = m
        self.n = n
        self.lr = lr
        self.weights = initial_weights if initial_weights is not None else np.random.uniform(-1, 1, (n, m))
        self.biases = initial_biases if initial_biases is not None else np.random.uniform(-1, 1, n)

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_biases(self) -> np.ndarray:
        return self.biases

    def get_lr(self) -> float:
        return self.lr

    def forward(self, x) -> np.ndarray:
        return self.weights @ x + self.biases