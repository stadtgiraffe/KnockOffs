import numpy as np
import scipy.special as sp
from operator import mul
from functools import reduce
from sklearn.preprocessing import StandardScaler

def load_IHDP_data(type_a=False, i=7):
    if type_a == True:
        path_data = "./IHDP"
        data_train = np.loadtxt(path_data + '/ihdp_npci_train_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)
        data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)
    else:
        path_data = "./IHDP_b"
        data_train = np.loadtxt(path_data + '/ihdp_npci_train_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)
        data_test = np.loadtxt(path_data + '/ihdp_npci_test_' + str(i + 1) + '.csv', delimiter=',', skiprows=1)

    t_train, y_train = data_train[:, 0], data_train[:, 1][:, np.newaxis]
    mu_0_train, mu_1_train, x_train = data_train[:, 3][:, np.newaxis], data_train[:, 4][:, np.newaxis], data_train[
                                                                                                        :, 5:]

    t_test, y_test = data_test[:, 0].astype('float32'), data_test[:, 1][:, np.newaxis].astype('float32')
    mu_0_test, mu_1_test, x_test = data_test[:, 3][:, np.newaxis].astype('float32'), data_test[:, 4][:,
                                                                                     np.newaxis].astype('float32'), \
                                   data_test[:, 5:].astype('float32')

    data_train = {'x': x_train, 't': t_train, 'y': y_train, 't': t_train, 'mu_0': mu_0_train, 'mu_1': mu_1_train}

    data_train['t'] = data_train['t'].reshape(-1,
                                              1)  # we're just padding one dimensional vectors with an additional dimension
    data_train['y'] = data_train['y'].reshape(-1, 1)
    # rescaling y between 0 and 1 often makes training of DL regressors easier
    data_train['y_scaler'] = StandardScaler().fit(data_train['y'])
    data_train['ys'] = data_train['y_scaler'].transform(data_train['y'])

    data_test = {'x': x_test, 't': t_test, 'y': y_test, 't': t_test, 'mu_0': mu_0_test, 'mu_1': mu_1_test}
    data_test['t'] = data_test['t'].reshape(-1,
                                            1)  # we're just padding one dimensional vectors with an additional dimension
    data_test['y'] = data_test['y'].reshape(-1, 1)
    # rescaling y between 0 and 1 often makes training of DL regressors easier
    data_test['y_scaler'] = StandardScaler().fit(data_test['y'])
    data_test['ys'] = data_test['y_scaler'].transform(data_test['y'])

    return data_train, data_test

def covariance_AR1(p, rho):
    """
    Construct the covariance matrix of a Gaussian AR(1) process
    """
    assert len(rho)>0, "The list of coupling parameters must have non-zero length"
    assert 0 <= max(rho) <= 1, "The coupling parameters must be between 0 and 1"
    assert 0 <= min(rho) <= 1, "The coupling parameters must be between 0 and 1"

    # Construct the covariance matrix
    Sigma = np.zeros(shape=(p,p))
    for i in range(p):
        for j in range(i,p):
            Sigma[i][j] = reduce(mul, [rho[l] for l in range(i,j)], 1)
    Sigma = np.triu(Sigma)+np.triu(Sigma).T-np.diag(np.diag(Sigma))
    return Sigma

def cov2cor(Sigma):
    """
    Converts a covariance matrix to a correlation matrix
    :param Sigma : A covariance matrix (p x p)
    :return: A correlation matrix (p x p)
    """
    sqrtDiagSigma = np.sqrt(np.diag(Sigma))
    scalingFactors = np.outer(sqrtDiagSigma,sqrtDiagSigma)
    Sigma = np.divide(Sigma, scalingFactors)
    Sigma[np.diag_indices(Sigma.shape[0])] = 1
    return Sigma

def scramble(a, axis=1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """
    b = np.random.random(a.shape)
    idx = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled

class GaussianAR1:
    """
    Gaussian AR(1) model
    """
    def __init__(self, p, rho):
        """
        Constructor
        :param p      : Number of variables
        :param rho    : A coupling parameter
        :return:
        """
        self.p = p
        self.rho = rho
        self.Sigma = covariance_AR1(self.p, [self.rho]*(self.p-1))
        self.mu = np.zeros((self.p,))

    def sample(self, n=1, **args):
        """
        Sample the observations from their marginal distribution
        :param n: The number of observations to be sampled (default 1)
        :return: numpy matrix (n x p)
        """
        return np.random.multivariate_normal(self.mu, self.Sigma, n)

class GaussianMixtureAR1:
    """
    Gaussian mixture of AR(1) model
    """

    def __init__(self, p, rho_list, proportions=None):
        # Dimensions
        self.p = p
        # Number of components
        self.K = len(rho_list)
        # Proportions for each Gaussian
        if(proportions==None):
            self.proportions = [1.0/self.K]*self.K
        else:
            self.proportions = proportions

        # Initialize Gaussian distributions
        self.normals = []
        self.Sigma = np.zeros((self.p,self.p))
        self.mu = np.zeros((self.p,))
        for k in range(self.K):
            rho = rho_list[k]
            self.normals.append(GaussianAR1(self.p, rho))
            self.Sigma += self.normals[k].Sigma / self.K

    def sample(self, n=1, **args):
        """
        Sample the observations from their marginal distribution
        :param n: The number of observations to be sampled (default 1)
        :return: numpy matrix (n x p)
        """
        # Sample vector of mixture IDs
        Z = np.random.choice(self.K, n, replace=True)
        # Sample multivariate Gaussians
        X = np.zeros((n,self.p))
        for k in range(self.K):
            k_idx = np.where(Z==k)[0]
            n_idx = len(k_idx)
            X[k_idx,:] = self.normals[k].sample(n_idx)
        return X

class SparseGaussian:
    """
    Uncorrelated but dependent variables
    """
    def __init__(self, p, m):
        self.p = p
        self.m = m
        # Compute true covariance matrix
        pAA = sp.binom(self.p-1, self.m-1) / sp.binom(self.p, self.m)
        pAB = sp.binom(self.p-2, self.m-2) / sp.binom(self.p, self.m)
        self.Sigma = np.eye(self.p)*pAA + (np.ones((self.p,self.p))-np.eye(self.p)) * pAB
        self.mu = np.zeros((self.p,))

    def sample(self, n=1, **args):
        L = np.repeat(np.reshape(range(self.p),(1,self.p)), n, axis=0)
        L = scramble(L)
        S = (L < self.m).astype("int")
        V = np.random.normal(0,1,(n,1))
        X = S * V
        return X

class MultivariateStudentT:
    """
    Multivariate Student's t distribution
    """
    def __init__(self, p, df, rho):
        assert df > 2, "Degrees of freedom must be > 2"
        self.p = p
        self.df = df
        self.rho = rho
        self.normal = GaussianAR1(p, rho)
        self.Sigma = self.normal.Sigma * self.df/(self.df-2.0)
        self.mu = np.zeros((self.p,))

    def sample(self, n=1, **args):
        '''
        Output:
        Produce M samples of d-dimensional multivariate t distribution
        Input:
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        n = # of samples to produce
        '''
        Z = self.normal.sample(n)
        G = np.tile(np.random.gamma(self.df/2.,2./self.df,n),(self.p,1)).T
        return Z/np.sqrt(G)

class DataSampler:
    """
    Model for the explanatory variables
    """
    def __init__(self, params, standardize=True):
        self.p = params['p']
        self.standardize = standardize

        self.model_name = params["model"]
        if(self.model_name=="gaussian"):
            self.model = GaussianAR1(self.p, params["rho"])
            self.name = "gaussian"
        elif(self.model_name=="gmm"):
            self.model = GaussianMixtureAR1(self.p, params["rho-list"])
            self.name = "gmm"
        elif(self.model_name=="sparse"):
            self.model = SparseGaussian(self.p, params["sparsity"])
            self.name = "sparse"
        elif(self.model_name=="mstudent"):
            self.model = MultivariateStudentT(self.p, params["df"], params["rho"])
            self.name = "mstudent"
        else:
            raise Exception('Unknown model: '+self.model_name)

        # Center and scale distribution parameters
        if(self.standardize):
            self.Sigma = cov2cor(self.model.Sigma)
            self.mu = 0 * self.model.mu
        else:
            self.Sigma = self.model.Sigma
            self.mu = self.model.mu

    def scaler(self, X):
        return (X-self.model.mu) / np.sqrt(np.diag(self.model.Sigma))

    def sample(self, n, **args):
        X = self.model.sample(n, **args)
        if(self.standardize):
            X = self.scaler(X)
        return X
