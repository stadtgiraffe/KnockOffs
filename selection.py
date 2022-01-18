import numpy as np
from glmnet_python.cvglmnet import cvglmnet
from glmnet_python.cvglmnetCoef import cvglmnetCoef
from sklearn import preprocessing


def kfilter(W, offset=1.0, q=0.5):
    """
    Adaptive significance threshold with the knockoff filter
    :param W: vector of knockoff statistics
    :param offset: equal to one for strict false discovery rate control
    :param q: nominal false discovery rate
    :return a threshold value for which the estimated FDP is less or equal q
    """
    t = np.insert(np.abs(W[W != 0]), 0, 0)
    t = np.sort(t)
    ratio = np.zeros(len(t))
    for i in range(len(t)):
        ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
    # print(f'\n{ratio}')
    index = np.where(ratio <= q)[0]
    if len(index) == 0:
        thresh = float('inf')
    else:
        thresh = t[index[0]]

    return thresh


def sample_Y(X, signal_n=20, signal_a=10.0):
    n,p = X.shape
    beta = np.zeros((p,1))
    beta_nonzero = np.random.choice(p, signal_n, replace=True)
    beta[beta_nonzero,0] = (2*np.random.choice(2,signal_n)-1) * signal_a / np.sqrt(n)
    y = np.dot(X,beta) + np.random.normal(size=(n,1))
    return y, beta


def lasso_stats(X, Xk, y, alpha=0.1, scale=True):
    X = X.astype("float")
    Xk = Xk.astype("float")
    p = X.shape[1]
    if scale:
        X_concat = preprocessing.scale(np.concatenate((X, Xk), 1))
    else:
        X_concat = np.concatenate((X,Xk),1)
    cols_order = np.random.choice(X_concat.shape[1],X_concat.shape[1],replace=False)
    cvfit = cvglmnet(x=X_concat[:,cols_order].copy(), y=y.copy(), family='gaussian', alpha=alpha)
    Z = np.zeros((2*p,))
    Z[cols_order] = cvglmnetCoef(cvfit, s='lambda_min').squeeze()[1:]
    W = np.abs(Z[0:p]) - np.abs(Z[p:(2*p)])
    return(W.squeeze())


def select(W, beta, nominal_fdr=0.1):
    W_threshold = kfilter(W, q=nominal_fdr)
    selected = np.where(W >= W_threshold)[0]
    nonzero = np.where(beta!=0)[0]
    TP = len(np.intersect1d(selected, nonzero))
    FP = len(selected) - TP
    FDP = FP / max(TP+FP, 1.0)
    POW = TP / max(len(nonzero), 1.0)
    return selected, FDP, POW


def select_IHDP(W, nominal_fdr=0.1):
    W_threshold = kfilter(W, q=nominal_fdr)
    print(f'\n {W_threshold}')
    selected = np.where(W >= W_threshold)[0]
    return selected
