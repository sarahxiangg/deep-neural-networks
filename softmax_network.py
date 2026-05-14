import numpy as np
from network import Network


class SoftmaxNetwork(Network):

    def calculate_CCE(self, probs: np.ndarray, true: int) -> float:
        eps = 1e-12
        return -np.log(probs[true] + eps)

    def backward(self, probs: np.ndarray, true: int) -> np.ndarray:
        target = self._make_target(probs, true)

        # softmax + cross entropy gradient
        grads = probs - target

        for layer in reversed(self.layers):
            grads = layer.backward(grads)

        return grads

    def average_MSE(self, data):
        if len(data) == 0:
            return 0.0

        total = 0.0
        for x, y in data:
            probs = self.forward(x)
            total += self.calculate_CCE(probs, y)

        return total / len(data)

    def test(self, dataset: dict):
        test_data = dataset["test"]

        if len(test_data) == 0:
            return 0.0, 0.0

        total_loss = 0.0
        correct = 0

        for x, y in test_data:
            probs = self.forward(x)
            total_loss += self.calculate_CCE(probs, y)

            if np.argmax(probs) == y:
                correct += 1

        return total_loss / len(test_data), correct / len(test_data)