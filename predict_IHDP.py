import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from Deep_Knockoffs.machine import KnockoffMachine
from Deep_Knockoffs.gaussian import GaussianKnockoffs
import data
import parameters
import selection
import sys
from data import load_IHDP_data

# Number of features
p = 25

# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution

# Number of training examples

data_train, data_test = load_IHDP_data(type_a=True, i=1)
# data_train['x'] = np.concatenate((data_train['x'], np.random.randn(data_train['x'].shape[0], 5)), 1)
# data_test['x'] = np.concatenate((data_test['x'], np.random.randn(data_test['x'].shape[0], 5)), 1)

k = 9
for k in range(1, k+1):
    ds_train, ds_test = load_IHDP_data(type_a=True, i=k)
    data_train['x'] = np.concatenate((data_train['x'], ds_train['x']), 0)
    data_train['t'] = np.concatenate((data_train['t'], ds_train['t']), 0)
    data_train['y'] = np.concatenate((data_train['y'], ds_train['y']), 0)
    data_train['mu_0'] = np.concatenate((data_train['mu_0'], ds_train['mu_0']), 0)
    data_train['mu_1'] = np.concatenate((data_train['mu_1'], ds_train['mu_1']), 0)

    data_test['x'] = np.concatenate((data_test['x'], ds_test['x']), 0)
    data_test['t'] = np.concatenate((data_test['t'], ds_test['t']), 0)
    data_test['y'] = np.concatenate((data_test['y'], ds_test['y']), 0)
    data_test['mu_0'] = np.concatenate((data_test['mu_0'], ds_test['mu_0']), 0)
    data_test['mu_1'] = np.concatenate((data_test['mu_1'], ds_test['mu_1']), 0)

model = "gaussian"
n = data_train['x'].shape[0]

# Compute the empirical covariance matrix of the training data
SigmaHat = np.cov(data_train['x'], rowvar=False)

# Initialize generator of second-order knockoffs
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data_train['x'], 0), method="equi")

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(corr_g))))

# Load the default hyperparameters for this model
training_params = parameters.GetTrainingHyperParams(model)

# Set the parameters for training deep knockoffs
pars = dict()
# Number of epochs
pars['epochs'] = 1000
# Number of iterations over the full data per epoch
pars['epoch_length'] = 10
# Data type, either "continuous" or "binary"
pars['family'] = "binary"
# Dimensions of the data
pars['p'] = p
# Size of the test set
pars['test_size'] = 1
# Batch size
pars['batch_size'] = int(0.1*n)
# Learning rate
pars['lr'] = 0.001
# When to decrease learning rate (unused when equal to number of epochs)
pars['lr_milestones'] = [pars['epochs']]
# Width of the network (number of layers is fixed to 6)
pars['dim_h'] = int(2*p)
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

# Where the machine is stored
checkpoint_name = "tmp/" + model

# Initialize the machine
machine = KnockoffMachine(pars)

# Load the machine
machine.load(checkpoint_name, ds=k)

# Amplitude of the non-zero coefficients
# signal_amplitude_vec = [1]

# Compute the FDR as the average proportion of false discoveries over n_experiments
n_experiments = 10

# Target FDR level
nominal_fdr = 0.3

test_params = parameters.GetFDRTestParams(model)

# Initialize table of results
results = pd.DataFrame(columns=['Model', 'Experiment', 'Method', 'FDP', 'Power', 'Amplitude', 'Signals', 'Alpha', 'FDR.nominal'])

# Run experiments
print("Running %d experiments:" % (n_experiments))

for exp_id in range(n_experiments):
    # Show progress
    sys.stdout.write('\r')
    sys.stdout.write(
        "[%-25s] %d%%" % ('=' * int((exp_id + 1) / n_experiments * 25), ((exp_id + 1) / n_experiments) * 100))
    sys.stdout.flush()

    # Generate deep knockoffs
    Xk_m = machine.generate(data_train['x'])
    # Compute importance statistics
    W_m = selection.lasso_stats(data_train['x'], Xk_m, np.array(data_train['y'], dtype=np.float64),
                                alpha=test_params["elasticnet_alpha"], scale=False)
    # Select important variables with the knockoff filter
    selected_m = selection.select_IHDP(W_m, nominal_fdr=nominal_fdr)
    # print(f'\n {W_m} \n {selected_m} \n')
    # print(f'    selected covariates: {selected_m}')
    print(f'\n {W_m}')


for exp_id in range(n_experiments):
    # Generate second-order knockoffs
    Xk_g = second_order.generate(data_test['x'])
    # Compute importance statistics
    W_g = selection.lasso_stats(data_test['x'], Xk_g, np.array(data_test['y'], dtype=np.float64),
                                alpha=test_params["elasticnet_alpha"], scale=False)
    # Select important variables with the knockoff filter
    selected_g = selection.select_IHDP(W_g, nominal_fdr=nominal_fdr)
    # print(f'\n {W_g} \n {selected_g} \n')
    # print(f'    selected covariates: {selected_g}')
    print(f'\n {W_g}')

sys.stdout.write('\n')

