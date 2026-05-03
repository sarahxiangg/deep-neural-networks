import numpy as np

class SigmoidLayer():
    def __init__(self, n: int):
        self.m = n
        self.n = n

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x  
        return 1 / (1 + np.exp(-x))