import numpy as np
import _pickle as cp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.stats import bernoulli

NUM_TESTS = 2
TEST_PERCENT_START = 10
TEST_PERCENT_END = 100
TEST_PERCENT_INC = 10


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


class NBCFeatureParamBinary(NBCFeatureParam):
    def __init__(self, feature_idx: int, theta: float):
        super().__init__(feature_idx)
        self._theta = theta
        self._prob = bernoulli(theta)

    def get_probability(self, val):
        return self._prob.pmf(val)



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

    def generate_binary_params(Xtrain, label_indices, binary_features_isx):
        feature_params_a = []
        mean_features = np.mean(Xtrain[label_indices, :], axis=0)
        for idx_feature, mean_feature in enumerate(mean_features):
            feature_param = NBCFeatureParamBinary(binary_features_isx[idx_feature],
                                                  mean_feature)
            feature_params_a.append(feature_param)

        return feature_params_a

    # TODO: Change so it adds zero elements
    def fit(self, Xtrain, ytrain):
        unique_labels, count_elements = np.unique(ytrain, return_counts=True)
        num_elements_d = dict(zip(unique_labels, count_elements))
        self._pi = {key: value/ytrain.size for key, value in num_elements_d.items()}

        label_feature_params = []
        real_features_idx = np.squeeze(np.argwhere(self._feature_types == 'r'), 1)
        binary_features_idx = np.squeeze(np.argwhere(self._feature_types == 'b'), 1)
        Xtrain_real_features = Xtrain[:, real_features_idx]
        Xtrain_binary_features = Xtrain[:, binary_features_idx]

        for label in unique_labels:
            label_indices = np.squeeze(np.argwhere(ytrain == label))
            feature_params_real = NBC.generate_real_params(Xtrain_real_features,
                                                           label_indices, real_features_idx)
            feature_params_bin = NBC.generate_binary_params(Xtrain_binary_features,
                                                          label_indices, binary_features_idx)
            # TODO: reorder features (arrays)
            label_feature_params.append(feature_params_bin + feature_params_real)
        self._label_feature_params = label_feature_params

    def get_features_cond_prob(self, label, x_new):
        label_feature_params = self._label_feature_params[label]
        features_prob = 1
        for idx, label_feature_param in enumerate(label_feature_params):
            features_prob = label_feature_param.get_probability\
                                (x_new[label_feature_param._feature_idx])\
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


def data_shuffle(X, y):
    N, D = X.shape
    Ntrain = int(0.8 * N)
    shuffler = np.random.permutation(N)
    Xtrain = X[shuffler[:Ntrain]]
    ytrain = y[shuffler[:Ntrain]]
    Xtest = X[shuffler[Ntrain:]]
    ytest = y[shuffler[Ntrain:]]
    return Xtrain, ytrain, Xtest, ytest


def load_data(is_iris: bool):
    if is_iris:
        iris = load_iris()
        X, y = iris['data'], iris['target']
    else:
        X, y = cp.load(open('voting.pickle', 'rb'))
    return X, y


def choose_and_test_data(X, y):
    Xtrain, ytrain, Xtest, ytest = data_shuffle(X, y)
    test_accuracies_nbc = []
    test_accuracies_lr = []
    N, D = Xtrain.shape
    for test_per in range(TEST_PERCENT_START, TEST_PERCENT_END + 1, TEST_PERCENT_INC):
        size = int(test_per / 100 * N)

        nbc = NBC(feature_types=['r'] * D, num_classes=2)
        nbc.fit(Xtrain[:size,:], ytrain[:size])
        yhat = nbc.predict(Xtest)
        test_accuracies_nbc.append(np.mean(yhat == ytest))

        lr = LogisticRegression(penalty='l2',C=1e1, solver='lbfgs',
                                multi_class='multinomial')
        lr.fit(Xtrain[:size, :], ytrain[:size])
        yhat = lr.predict(Xtest)
        test_accuracies_lr.append(np.mean(yhat == ytest))
    return np.array(test_accuracies_nbc), np.array(test_accuracies_lr)


if __name__ == "__main__":
    X, y = load_data(is_iris=True)
    first = True
    for num_test in range(NUM_TESTS):
        test_acc_nbc, test_acc_lr = choose_and_test_data(X, y)
        test_acc_nbc_a, test_acc_lr_a = (test_acc_nbc, test_acc_lr) \
            if first else (np.vstack((test_acc_nbc_a, test_acc_nbc)), \
                                    np.vstack((test_acc_lr_a, test_acc_lr)))
        first = False
        print("Test advance {}".format((num_test+1)/NUM_TESTS))
    test_acc_nbc = test_acc_nbc_a.mean(axis=0)
    test_acc_lr = test_acc_lr_a.mean(axis=0)
    print("Accuracy advance NBC:")
    print(test_acc_nbc)
    print("Accuracy advance LR")
    print(test_acc_lr)
    fig = plt.figure(0)
    plt.plot(np.arange(TEST_PERCENT_START, TEST_PERCENT_END + 1, TEST_PERCENT_INC),
             test_acc_nbc, 'r--',
             np.arange(TEST_PERCENT_START, TEST_PERCENT_END + 1, TEST_PERCENT_INC),
             test_acc_lr, 'b-')
    plt.gca().legend(('Naive Bayes classifier', 'Logistic regression'))
    plt.xlabel('Percent of training set')
    plt.ylabel('Average classification error')
    plt.show()

