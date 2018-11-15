import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_iris


class NBC:
    def __init__(self, feature_types: list, num_classes: int = 4):
        self._feature_types = feature_types
        self._num_classes =num_classes
        self._pi = []

    # TODO: Change so it adds zero elements
    def fit(self, Xtrain, ytrain):
        unique_elements, count_elements = np.unique(ytrain, return_counts=True)
        print(unique_elements)
        print(count_elements)
        num_elements_d = dict(zip(unique_elements, count_elements))
        print(ytrain.size)
        self._pi = {key: value/ytrain.size for key, value in num_elements_d.items()}
        print(self._pi)

    def predict(self, XTest):
        pass


if __name__ == "__main__":
    print("Hello world")
    iris = load_iris()
    X, y = iris['data'], iris['target']
    X, y = cp.load(open('voting.pickle', 'rb'))
    nbc = NBC(feature_types=['b', 'r', 'b'], num_classes=4)
    nbc.fit(X, y)


