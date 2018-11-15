import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import load_iris
from abc import ABC, abstractmethod
from scipy.stats import norm


class NBCFeatureParam(ABC):
    def __init__(self, feature_idx: int):
        self._feature_idx = feature_idx

    @abstractmethod
    def get_probability(self, val):
        pass


'''
Class used for storing parameters about features of specific class.
'''
class NBCFeatureParamReal(NBCFeatureParam):
    def __init__(self, feature_idx: int, mean: float, std: float):
        super().__init__(feature_idx)
        self._mean = mean
        self._std = std
        self._prob = norm(mean, std)

    def get_probability(self, val):
        return self._prob.pdf(val)


def data_shuffle(X, y):
    N, D = X.shape
    Ntrain = int(0.8 * N)
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]
    return Xtrain, ytrain, Xtest, ytest

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
        self._label_feature_params = label_feature_params

    def get_features_cond_prob(self, label, x_new):
        label_feature_params = self._label_feature_params[label]
        features_prob = 1
        for idx, label_feature_param in enumerate(label_feature_params):
            features_prob = label_feature_param.get_probability(x_new[idx])\
                            * features_prob
        return features_prob

    def get_cond_prob(self, label, x_new):
        return self._pi[label] * self.get_features_cond_prob(label, x_new)

    def get_max_cond_prob(self, x_new):
        cur_max_prob = 0
        max_prob_label = -1
        for label in range(0, self._num_classes):
            cur_prob = self.get_cond_prob(label, x_new)
            if cur_prob >= cur_max_prob:
                max_prob_label = label
                cur_max_prob = cur_prob
        return max_prob_label

    def predict(self, Xtest):
        N, D = Xtest.shape
        ytest = []
        for idx in range(0, N):
            ytest.append(self.get_max_cond_prob(np.squeeze(Xtest[idx, :])))
        return np.array(ytest)


if __name__ == "__main__":
    print("Hello world")
    iris = load_iris()
    X, y = iris['data'], iris['target']
    #X, y = cp.load(open('voting.pickle', 'rb'))
    Xtrain, ytrain, Xtest, ytest = data_shuffle(X, y);
    nbc = NBC(feature_types=['r', 'r', 'r', 'r'], num_classes=3)
    nbc.fit(Xtrain, ytrain)
    yhat = nbc.predict(Xtest)
    print(yhat)
    print(ytest)
    test_accuracy = np.mean(yhat == ytest)
    print(test_accuracy)


