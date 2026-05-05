import numpy as np

class Network():
    def __init__(self, layers):
        self.layers = layers
        self.last_output = None

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        self.last_output = out
        return out

    def get_layers(self):
        return self.layers

    def _make_target(self, logits: np.ndarray, true: int) -> np.ndarray:
        target = np.zeros_like(logits)
        if 0 <= true < len(logits):
            target[true] = 1
        return target

    def calculate_MSE(self, logits: np.ndarray, true: int) -> float:
        target = self._make_target(logits, true)
        return np.sum((logits - target) ** 2) / 2

    def backward(self, logits: np.ndarray, true: int) -> np.ndarray:
        if self.last_output is None:
            return np.zeros_like(logits)
        target = self._make_target(logits, true)
        grads = logits - target
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def average_MSE(self, data):
        if len(data) == 0:
            return 0.0
        total = 0.0
        for x, y in data:
            logits = self.forward(x)
            total += self.calculate_MSE(logits, y)
        return total / len(data)

    def test(self, dataset: dict):
        test_data = dataset["test"]
        if len(test_data) == 0:
            return 0.0, 0.0
        total_loss = 0.0
        correct = 0
        for x, y in test_data:
            logits = self.forward(x)
            total_loss += self.calculate_MSE(logits, y)
            if np.argmax(logits) == y:
                correct += 1
        return total_loss / len(test_data), correct / len(test_data)

    def train(self, dataset: dict, max_epochs: int, validation_interval: int = 0) -> int:
        train_data = dataset["train"]
        valid_data = dataset.get("valid", [])

        best_params = []
        best_val_error = float("inf")

        for layer in self.layers:
            if hasattr(layer, "get_weights"):
                best_params.append((layer.get_weights().copy(), layer.get_biases().copy()))
            else:
                best_params.append(None)

        for epoch in range(max_epochs):
            for x, y in train_data:
                logits = self.forward(x)
                self.backward(logits, y)

            if validation_interval > 0 and (epoch + 1) % validation_interval == 0:
                val_error = self.average_MSE(valid_data)

                if val_error > best_val_error:
                    for layer, params in zip(self.layers, best_params):
                        if params is not None:
                            w, b = params
                            layer.weights = w.copy()
                            layer.biases = b.copy()
                    return epoch + 1

                best_val_error = val_error
                best_params = []
                for layer in self.layers:
                    if hasattr(layer, "get_weights"):
                        best_params.append((layer.get_weights().copy(), layer.get_biases().copy()))
                    else:
                        best_params.append(None)

        return max_epochs
