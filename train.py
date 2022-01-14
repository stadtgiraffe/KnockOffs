import numpy as np
from Deep_Knockoffs.machine import KnockoffMachine
from Deep_Knockoffs.gaussian import GaussianKnockoffs
import data
import parameters
from data import load_IHDP_data

data_train, data_test = load_IHDP_data(type_a=True, i=0)

# Compute the empirical covariance matrix of the training data
SigmaHat = np.cov(data_train['x'], rowvar=False)

# Initialize generator of second-order knockoffs
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data_train['x'], 0), method="sdp")

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(corr_g))))

# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
model = "sparse"
# Number of features
n = data_train['x'].shape[0]
p = data_train['x'].shape[1]

# Load the default hyperparameters for this model
training_params = parameters.GetTrainingHyperParams(model)

# Set the parameters for training deep knockoffs
pars = dict()
# Number of epochs
pars['epochs'] = 1000
# Number of iterations over the full data per epoch
pars['epoch_length'] = 100
# Data type, either "continuous" or "binary"
pars['family'] = "continuous"
# Dimensions of the data
pars['p'] = p
# Size of the test set
pars['test_size'] = 25
# Batch size
pars['batch_size'] = int(0.5*n)
# Learning rate
pars['lr'] = 0.001
# When to decrease learning rate (unused when equal to number of epochs)
pars['lr_milestones'] = [pars['epochs']]
# Width of the network (number of layers is fixed to 6)
pars['dim_h'] = int(10*p)
# Penalty for the MMD distance
pars['GAMMA'] = training_params['GAMMA']
# Penalty encouraging second-order knockoffs
pars['LAMBDA'] = training_params['LAMBDA']
# Decorrelation penalty hyperparameter
pars['DELTA'] = training_params['DELTA']
# Target pairwise correlations between variables and knockoffs
pars['target_corr'] = corr_g
# Kernel widths for the MMD measure (uniform weights)
pars['alphas'] = [1., 2., 4., 8., 16., 32., 64., 128.]

# Where to store the machine
checkpoint_name = "tmp/" + model

# Where to print progress information
logs_name = "tmp/" + model + "_progress.txt"

# Initialize the machine
machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

# Train the machine
print("Fitting the knockoff machine...")
machine.train(data_train['x'])
