import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def encoder(Y):
    if isinstance(Y, (pd.core.series.Series, np.ndarray)):
        # number of classifications
        try:
            Y = Y.flatten()
        except:
            pass
        labels = np.unique(Y)
        n = labels.shape[0]
        m = Y.shape[0]
    elif isinstance(Y, (list, tuple)):
        labels = list(set(Y))
        n = len(labels)
        m = len(Y)

    else:
        raise Exception('wrong type')

    classifier_matrix = np.ones([m, n], dtype=object)
    i = 0
    for col in classifier_matrix.T:
        col = col*labels[i]
        encode = (col == Y)*1
        classifier_matrix.T[i] = encode
        i += 1
    try:
        classifier_matrix = classifier_matrix.astype(int)
    except ValueError:
        pass
    return classifier_matrix

def data_normalize(X):
    X_min = np.min(X, 0)
    X_max = np.max(X, 0)
    X_norm = (X-X_min) / (X_max - X_min)
    return X_norm

class N_Standardise:
    def __init__(self, X):
        self.mu = np.mean(X,0)
        self.sigma = np.std(X,0)
        self.X = X

    def normalize(self):
        self.X = (self.X-self.mu)/self.sigma
        return self.X

    def convert(self,X):
        X = (X-self.mu)/self.sigma
        return X

def poly_features(X,d=2):
    poly = PolynomialFeatures(degree=d)
    transformed = poly.fit_transform(X)
    transformed = np.delete(transformed, 0, axis=1)
    return transformed

def data_standardise(X):
    mu = X - np.mean(X, 0)
    sigma = np.std(X, 0)
    X = mu/sigma
    return X


def min_max(X):
    N = (X - np.min(X,0)) / (np.max(X,0)-np.min(X,0))
    print('min:',np.min(X,0))
    print('diff:',np.max(X,0)-np.min(X,0))
    return N


def check_y(y):
    if isinstance(y, (list)):
        y = np.asarray(y)
        y = y.reshape(y.shape[0], 1)  # shape as column vector
    elif isinstance(y, (pd.core.series.Series, pd.DataFrame)):
        y = y.to_numpy()
        y = y.reshape(y.shape[0], 1)  # shape as column vector
    elif isinstance(y, (np.ndarray)):
        y = y.reshape(-1, 1)
    return y


def check_x(X):
    if isinstance(X, (list)):
        X = np.asarray(X)
    elif isinstance(X, (pd.core.series.Series, pd.DataFrame)):
        X = X.to_numpy()
    m = X.shape[0]
    try:
        n = X.shape[1]
    except IndexError:
        X = X.reshape(1, X.shape[0])  # reshape as row vector
        n = X.shape[0]
    return (X, n, m)


def multinomial_check(y):
    if isinstance(y, (pd.core.series.Series, pd.DataFrame, np.ndarray)):
        labels = np.unique(y)
        n_labels = labels.shape[0]
        # m = y.shape[0]
    elif isinstance(y, (list, tuple)):
        labels = list(set(y))
        n_labels = len(labels)
        # m = len(y)
    else:
        # raise Exception('wrong type')
        pass
    if n_labels >= 2:
        return (labels, n_labels)
    else:
        return False


def matdot(X, Y, verbose=False):
    '''if X or Y is single dimension'''
    '''X will always be shape to be a row vector '''
    '''Y will be shaped according to X shape for multiplication'''
    if X.ndim == 1:
        X = X.reshape(1, X.shape[0])
    if Y.ndim == 1:
        if X.shape[1] == 1:
            Y = Y.reshape(1,Y.shape[0])
        else:
            Y = Y.reshape(Y.shape[0],1)
    try:
        result = X@Y
        if verbose:
            print('X shape: {} '.format(X.shape))
            print('Y shape: {}'.format(Y.shape))
            print('Output shape{}'.format(result.shape))
        return result
    except ValueError:
        print('Fail to multiply. Shape is wrong')
        print('X shape: {} '.format(X.shape))
        print('Y shape: {}'.format(Y.shape))
        print(X)
        print(Y)
        return None
