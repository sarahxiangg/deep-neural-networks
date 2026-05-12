import csv
import random
import numpy as np
from itertools import product

from network import Network
from feedforward import FeedforwardLayer
from sigmoid import SigmoidLayer
from feedforward_with_momentum import FeedforwardLayer as MomentumFeedforwardLayer

np.random.seed(3608)
random.seed(3608)


def load_data(filename):
    data = []
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            x = np.array([float(v) for v in row[:-1]])
            y = 1 if row[-1].strip().lower() == "yes" else 0
            data.append((x, y))
    return data


def normalise(train, valid, test):
    X_train = np.array([x for x, y in train])
    col_min = X_train.min(axis=0)
    col_max = X_train.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1

    def scale(data):
        return [(( x - col_min) / col_range, y) for x, y in data]

    return scale(train), scale(valid), scale(test)


def evaluate(model, data):
    tp = tn = fp = fn = 0
    for x, y in data:
        output = model.forward(x)
        prediction = 1 if output[0] >= 0.5 else 0
        if prediction == 1 and y == 1:   tp += 1
        elif prediction == 0 and y == 0: tn += 1
        elif prediction == 1 and y == 0: fp += 1
        elif prediction == 0 and y == 1: fn += 1

    accuracy  = (tp + tn) / len(data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)
    return accuracy, precision, recall, f1, tp, tn, fp, fn



def build_standard(input_size, hidden_neurons, lr):
    return Network([
        FeedforwardLayer(input_size, hidden_neurons, lr),
        SigmoidLayer(hidden_neurons),
        FeedforwardLayer(hidden_neurons, 1, lr),
        SigmoidLayer(1)
    ])


def build_momentum(input_size, hidden_neurons, lr, momentum):
    return Network([
        MomentumFeedforwardLayer(input_size, hidden_neurons, lr, momentum=momentum),
        SigmoidLayer(hidden_neurons),
        MomentumFeedforwardLayer(hidden_neurons, 1, lr, momentum=momentum),
        SigmoidLayer(1)
    ])


def grid_search(train_data, valid_data, test_data, mode="standard"):
    input_size = len(train_data[0][0])

    learning_rates   = [0.01, 0.05, 0.1, 0.2]
    hidden_sizes     = [8, 16, 32, 64]
    max_epochs_list  = [200, 500]
    momentum_values  = [0.5, 0.8, 0.9, 0.95]  # only used in momentum mode

    dataset = {"train": train_data, "valid": valid_data, "test": test_data}

    best_f1     = -1
    best_config = None
    best_model  = None
    total       = 0

    if mode == "standard":
        combos = list(product(learning_rates, hidden_sizes, max_epochs_list))
        total  = len(combos)
        print(f"\nGrid search [{mode}]: {total} combinations\n")

        for i, (lr, hidden, max_ep) in enumerate(combos, 1):
            np.random.seed(3608)
            model = build_standard(input_size, hidden, lr)
            model.train(dataset, max_epochs=max_ep, validation_interval=10)
            _, _, _, f1, *_ = evaluate(model, valid_data)

            print(f"  [{i:>3}/{total}] lr={lr}  hidden={hidden:>2}  "
                  f"epochs={max_ep}  →  val F1={f1:.4f}")

            if f1 > best_f1:
                best_f1     = f1
                best_config = dict(lr=lr, hidden=hidden, max_epochs=max_ep)
                best_model  = model

    else:  # momentum
        combos = list(product(learning_rates, hidden_sizes,
                               max_epochs_list, momentum_values))
        total  = len(combos)
        print(f"\nGrid search [{mode}]: {total} combinations\n")

        for i, (lr, hidden, max_ep, mom) in enumerate(combos, 1):
            np.random.seed(3608)
            model = build_momentum(input_size, hidden, lr, mom)
            model.train(dataset, max_epochs=max_ep, validation_interval=10)
            _, _, _, f1, *_ = evaluate(model, valid_data)

            print(f"  [{i:>3}/{total}] lr={lr}  hidden={hidden:>2}  "
                  f"epochs={max_ep}  momentum={mom}  →  val F1={f1:.4f}")

            if f1 > best_f1:
                best_f1     = f1
                best_config = dict(lr=lr, hidden=hidden,
                                   max_epochs=max_ep, momentum=mom)
                best_model  = model

    return best_config, best_model



train_raw = load_data("train.csv")
valid_raw = load_data("validation.csv")
test_raw  = load_data("test.csv")

train_data, valid_data, test_data = normalise(train_raw, valid_raw, test_raw)

std_config, std_model = grid_search(train_data, valid_data, test_data,
                                     mode="standard")

acc, prec, rec, f1, tp, tn, fp, fn = evaluate(std_model, test_data)

print("\n" + "="*50)
print("BEST STANDARD MLP")
print("="*50)
print("Config:", std_config)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Confusion Matrix")
print(f"[[TN: {tn}  FP: {fp}]]")
print(f"[[FN: {fn}  TP: {tp}]]")

# --- momentum MLP ---
mom_config, mom_model = grid_search(train_data, valid_data, test_data,
                                     mode="momentum")

acc, prec, rec, f1, tp, tn, fp, fn = evaluate(mom_model, test_data)

print("\n" + "="*50)
print("BEST MOMENTUM MLP")
print("="*50)
print("Config:", mom_config)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Confusion Matrix")
print(f"[[TN: {tn}  FP: {fp}]]")
print(f"[[FN: {fn}  TP: {tp}]]")