import csv
import random
import argparse
import numpy as np
from itertools import product

# import custom neural network components
from network import Network
from feedforward import FeedforwardLayer
from sigmoid import SigmoidLayer
from feedforward_with_momentum import FeedforwardLayer as MomentumFeedforwardLayer
from softmax import SoftmaxLayer

# fix random seeds so results are reproducible
np.random.seed(3608)
random.seed(3608)


def load_data(filename):
    # load csv rows into feature vectors and binary labels
    data = []

    with open(filename, "r") as file:
        reader = csv.reader(file)

        # skip header row
        next(reader)

        for row in reader:
            # convert feature values to floats
            x = np.array([float(v) for v in row[:-1]])

            # convert class label into binary value
            y = 1 if row[-1].strip().lower() == "yes" else 0

            data.append((x, y))

    return data


def normalise(train, valid, test):
    # only use training data to calculate min and max values
    # this avoids leaking validation/test information into training
    X_train = np.array([x for x, y in train])

    col_min = X_train.min(axis=0)
    col_max = X_train.max(axis=0)
    col_range = col_max - col_min

    # avoid division by zero for constant columns
    col_range[col_range == 0] = 1

    def scale(data):
        # apply min-max normalisation to each feature column
        return [((x - col_min) / col_range, y) for x, y in data]

    return scale(train), scale(valid), scale(test)


def evaluate(model, data, algorithm="gradient"):
    # store confusion matrix counts
    tp = tn = fp = fn = 0

    for x, y in data:
        output = model.forward(x)

        # softmax outputs two class probabilities, so choose highest probability
        if algorithm == "softmax":
            prediction = int(np.argmax(output))

        # sigmoid output gives one probability for the positive class
        else:
            prediction = 1 if output[0] >= 0.5 else 0

        # update confusion matrix values
        if prediction == 1 and y == 1:
            tp += 1
        elif prediction == 0 and y == 0:
            tn += 1
        elif prediction == 1 and y == 0:
            fp += 1
        elif prediction == 0 and y == 1:
            fn += 1

    # calculate evaluation metrics
    accuracy = (tp + tn) / len(data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return accuracy, precision, recall, f1, tp, tn, fp, fn


def build_gradient(input_size, hidden_neurons, lr):
    # build standard mlp using sigmoid hidden and output layers
    return Network([
        FeedforwardLayer(input_size, hidden_neurons, lr),
        SigmoidLayer(hidden_neurons),
        FeedforwardLayer(hidden_neurons, 1, lr),
        SigmoidLayer(1)
    ])


def build_softmax(input_size, hidden_neurons, lr):
    # build mlp with two output neurons and softmax final layer
    return Network([
        FeedforwardLayer(input_size, hidden_neurons, lr),
        SigmoidLayer(hidden_neurons),
        FeedforwardLayer(hidden_neurons, 2, lr),
        SoftmaxLayer(2)
    ])


def build_model(input_size, hidden_neurons, lr, algorithm):
    # choose which model type to build
    if algorithm == "gradient":
        return build_gradient(input_size, hidden_neurons, lr)

    if algorithm == "softmax":
        return build_softmax(input_size, hidden_neurons, lr)

    raise ValueError("algorithm must be either 'gradient' or 'softmax'")


def grid_search(train_data, valid_data, test_data, algorithm="gradient"):
    # number of input features in the dataset
    input_size = len(train_data[0][0])

    # hyperparameter values to test
    learning_rates = [0.01, 0.05, 0.1, 0.2]
    hidden_sizes = [8, 16, 32, 64]
    max_epochs_list = [200, 500]

    # package data in the format expected by network.train()
    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }

    # track the best validation result
    best_f1 = -1
    best_config = None
    best_model = None

    # create every combination of hyperparameters
    combos = list(product(learning_rates, hidden_sizes, max_epochs_list))

    print(f"\nGrid search [{algorithm}]: {len(combos)} combinations\n")

    for i, (lr, hidden, max_ep) in enumerate(combos, 1):
        # reset seed so each configuration starts fairly
        np.random.seed(3608)

        # build and train model with current hyperparameters
        model = build_model(input_size, hidden, lr, algorithm)
        model.train(dataset, max_epochs=max_ep, validation_interval=10)

        # evaluate on validation data and select by f1-score
        _, _, _, f1, *_ = evaluate(model, valid_data, algorithm)

        print(
            f"  [{i:>3}/{len(combos)}] "
            f"lr={lr} hidden={hidden:>2} epochs={max_ep} "
            f"→ val F1={f1:.4f}"
        )

        # save model if it has the best validation f1 so far
        if f1 > best_f1:
            best_f1 = f1
            best_config = {
                "algorithm": algorithm,
                "lr": lr,
                "hidden": hidden,
                "max_epochs": max_ep
            }
            best_model = model

    return best_config, best_model


def main():
    # allow user to choose standard or softmax model from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=["gradient", "softmax"],
        default="gradient",
        help="Choose which training algorithm/model output to use"
    )

    args = parser.parse_args()

    # load pre-split dataset files
    train_raw = load_data("train.csv")
    valid_raw = load_data("validation.csv")
    test_raw = load_data("test.csv")

    # normalise all data using training set statistics
    train_data, valid_data, test_data = normalise(train_raw, valid_raw, test_raw)

    # find best hyperparameter configuration
    best_config, best_model = grid_search(
        train_data,
        valid_data,
        test_data,
        algorithm=args.algorithm
    )

    # evaluate best model on unseen test data
    acc, prec, rec, f1, tp, tn, fp, fn = evaluate(
        best_model,
        test_data,
        algorithm=args.algorithm
    )

    # print final test results
    print("\n" + "=" * 50)
    print(f"BEST {args.algorithm.upper()} MODEL")
    print("=" * 50)
    print("Config:", best_config)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix")
    print(f"[[TN: {tn}  FP: {fp}]]")
    print(f"[[FN: {fn}  TP: {tp}]]")


if __name__ == "__main__":
    main()