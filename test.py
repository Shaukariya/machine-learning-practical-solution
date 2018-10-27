import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt


def expand_with_ones(X):
    X_out = np.ones((X.shape[0], X.shape[1] + 1))
    X_out[:, 1:] = X
    return X_out


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
    plt.bar(unique, counts)
    plt.show()


def predict_simple(y_test):
    return y_test.mean()


def test_data(X_test, y_test, predictor: callable=None):
    # Applies function over rows.
    y_predicted = np.apply_along_axis(predictor, 1, X_test)
    mse = np.mean(np.square(np.subtract(y_predicted, y_test)))
    print("Mean squared error is {}".format(mse))


def standardize_data(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # By default arithmetic operations work when you have A r b, it will
    # r each row of A with row of b.
    X_std = (X - mean) / std
    return X_std, mean, std


# Computes parameters using the closed form solution.
def least_squares_compute_parameters(X_in, y):
    X = expand_with_ones(X_in)
    T = np.matmul(X.transpose(), X)
    T = np.linalg.inv(T)
    T = np.matmul(T, X.transpose())
    T = np.matmul(T, y)
    return T


def predict_linear_model(x, w):
    return np.dot(x, w)


if __name__ == "__main__":
    X, y = load_data()
    X_train, y_train, X_test, y_test = split_data(X, y)
    plot_bar_chart_score(X_train, y_train)
    print("Average is {}".format(predict_simple(y_train)))
    test_data(X_test, y_test, lambda x: predict_simple(y_train))

    print("Average")
    X_train_std, mean, std = standardize_data(X_train)
    w = least_squares_compute_parameters(X_train_std, y_train)

    print("Training linear model")
    test_data(expand_with_ones(X_train_std), y_train,
              lambda x: predict_linear_model(x, w))

    print("Testing linear model")
    X_test_std = (X_test - mean) / std
    test_data(expand_with_ones(X_test_std), y_test,
              lambda x: predict_linear_model(x, w))

    m = 1