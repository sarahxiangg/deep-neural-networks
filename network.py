import numpy as np

class Network():
    def __init__(self, layers: list):
        self.layers = layers

    def get_layers(self) -> list:
        return self.layers

    def validate_network(self):
        for i in range(len(self.layers) - 1):
            if self.layers[i].n != self.layers[i + 1].m:
                return False
        return True

    def forward(self, x) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def classify(self, x):
        output = self.forward(x)
        return np.argmax(output)