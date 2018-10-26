import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt


def load_data():
    X, y = cp.load(open('winequality-white.pickle', 'rb'))
    return X, y


# Arrays need to have the same length.
def split_data(X, y):
    N, _ = X.shape
    N_train = int(0.8 * N)
    X_train = X[:N_train]
    y_train = y[:N_train]
    X_test = X[N_train:]
    y_test = y[N_train:]
    return X_train, y_train, X_test, y_test


def plot_bar_chart_score(X_train, y_train):
    fix, ax = plt.subplots()
    unique, counts = np.unique(y_train, return_counts=True)
    print(counts)
    plt.bar(unique, counts)
    plt.show()


def predict_simple(X_train, y_train):
    return y_train.mean()


if __name__ == "__main__":
    X, y = load_data()
    X_train, y_train, X_test, y_test = split_data(X, y)
    plot_bar_chart_score(X_train, y_train)
    print("Score is {}".format(predict_simple(X_train, y_train)))
    m = 1