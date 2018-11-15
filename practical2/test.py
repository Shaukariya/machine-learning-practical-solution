import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_iris
from abc import ABC


class NBCFeatureParam(ABC):
    def __init__(self, feature_idx: int):
        self._feature_idx = feature_idx


'''
Class used for storing parameters about features of specific class.
'''
class NBCFeatureParamReal(NBCFeatureParam):
    def __init__(self, feature_idx: int, mean: float, std: float):
        super().__init__(feature_idx)
        self._mean = mean
        self._std = std


class NBC:
    def __init__(self, feature_types: list, num_classes: int = 4):
        self._feature_types = np.array(feature_types)
        self._num_classes =num_classes
        self._pi = []

    @staticmethod
    def generate_real_params(Xtrain, label_indices, real_features_idx):
        feature_params_a = []
        mean_features = np.mean(Xtrain[label_indices, :], axis=0)
        std_dev_features = np.std(Xtrain[label_indices, :], axis=0)
        for idx_feature, mean_feature in enumerate(mean_features):
            std_def_feature = std_dev_features[idx_feature]
            feature_param = NBCFeatureParamReal(real_features_idx[idx_feature],
                                                mean_feature, std_def_feature)
            feature_params_a.append(feature_param)

        return feature_params_a

    # TODO: Change so it adds zero elements
    def fit(self, Xtrain, ytrain):
        unique_labels, count_elements = np.unique(ytrain, return_counts=True)
        print(unique_labels)
        print(count_elements)
        num_elements_d = dict(zip(unique_labels, count_elements))
        print(ytrain.size)
        self._pi = {key: value/ytrain.size for key, value in num_elements_d.items()}
        print(self._pi)

        label_feature_params = []
        real_features_idx = np.squeeze(np.argwhere(self._feature_types == 'r'))
        Xtrain_real_features = Xtrain[:, real_features_idx]

        for label in unique_labels:
            label_indices = np.squeeze(np.argwhere(ytrain == label))
            feature_params_a = NBC.generate_real_params(Xtrain_real_features,
                                                        label_indices, real_features_idx)
            label_feature_params.append(feature_params_a)
        print("he")






    def predict(self, XTest):
        pass


if __name__ == "__main__":
    print("Hello world")
    iris = load_iris()
    X, y = iris['data'], iris['target']
    #X, y = cp.load(open('voting.pickle', 'rb'))
    nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
    nbc.fit(X, y)


