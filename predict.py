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
# model = "mstudent"
# distribution_params = parameters.GetDistributionParams(model, p)

# Number of training examples
data_train, data_test = load_IHDP_data(type_a=True, i=0)

model = "gaussian"
n = data_train['x'].shape[0]

# Sample training data
# X_train = DataSampler.sample(n)
print("Generated a training dataset of size: %d x %d." % data_train['x'].shape)

# Compute the empirical covariance matrix of the training data
SigmaHat = np.cov(data_train['x'], rowvar=False)

# Initialize generator of second-order knockoffs
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data_train['x'], 0), method="sdp")

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(corr_g))))

# Load the default hyperparameters for this model
training_params = parameters.GetTrainingHyperParams(model)

# Set the parameters for training deep knockoffs
pars = dict()
# Number of epochs
pars['epochs'] = 100
# Number of iterations over the full data per epoch
pars['epoch_length'] = 10
# Data type, either "continuous" or "binary"
pars['family'] = "continuous"
# Dimensions of the data
pars['p'] = p
# Size of the test set
pars['test_size'] = 0
# Batch size
pars['batch_size'] = int(0.5*n)
# Learning rate
pars['lr'] = 0.01
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

# Where the machine is stored
checkpoint_name = "tmp/" + model

# Initialize the machine
machine = KnockoffMachine(pars)

# Load the machine
machine.load(checkpoint_name, ds)

# Number of non-zero coefficients in P(Y|X)
signal_n = 25

# Amplitude of the non-zero coefficients
signal_amplitude_vec = [3, 5, 7, 10, 15, 20, 25]

# Compute the FDR as the average proportion of false discoveries over n_experiments
n_experiments = 100

# Target FDR level
nominal_fdr = 0.2

test_params = parameters.GetFDRTestParams(model)

# Initialize table of results
results = pd.DataFrame(columns=['Model', 'Experiment', 'Method', 'FDP', 'Power', 'Amplitude', 'Signals', 'Alpha', 'FDR.nominal'])

# Run experiments
for amp_id in range(len(signal_amplitude_vec)):
    # Set the signal amplitude
    signal_amplitude = signal_amplitude_vec[amp_id]
    print("Running %d experiments with signal amplitude: %.2f" % (n_experiments, signal_amplitude))

    for exp_id in range(n_experiments):
        # Show progress
        sys.stdout.write('\r')
        sys.stdout.write(
            "[%-25s] %d%%" % ('=' * int((exp_id + 1) / n_experiments * 25), ((exp_id + 1) / n_experiments) * 100))
        sys.stdout.flush()

        # Sample Y|X
        y, theta = selection.sample_Y(data_test['x'], signal_n=signal_n, signal_a=signal_amplitude)

        # Generate deep knockoffs
        Xk_m = machine.generate(data_test['x'])
        # Compute importance statistics
        W_m = selection.lasso_stats(data_test['x'], Xk_m, y, alpha=test_params["elasticnet_alpha"], scale=False)
        # Select important variables with the knockoff filter
        selected_m, FDP_m, POW_m = selection.select(W_m, theta, nominal_fdr=nominal_fdr)
        # Store results
        results = results.append({'Model': model, 'Experiment': exp_id, 'Method': 'deep',
                                  'Power': POW_m, 'FDP': FDP_m,
                                  'Amplitude': signal_amplitude, 'Signals': signal_n,
                                  'Alpha': 0.1, 'FDR.nominal': nominal_fdr}, ignore_index=True)

        # Generate second-order knockoffs
        Xk_g = second_order.generate(data_test['x'])
        # Compute importance statistics
        W_g = selection.lasso_stats(data_test['x'], Xk_g, y, alpha=test_params["elasticnet_alpha"], scale=False)
        # Select important variables with the knockoff filter
        selected_g, FDP_g, POW_g = selection.select(W_g, theta, nominal_fdr=nominal_fdr)
        print(f'\n {W_g} \n {selected_g} \n')
        print(selected_g)

        # Store results
        results = results.append({'Model': model, 'Experiment': exp_id, 'Method': 'second-order',
                                  'Power': POW_g, 'FDP': FDP_g,
                                  'Amplitude': signal_amplitude, 'Signals': signal_n,
                                  'Alpha': 0.1, 'FDR.nominal': nominal_fdr}, ignore_index=True)

    sys.stdout.write('\n')

# Summarize results
results.groupby(['Model', 'Method', 'Amplitude', 'Alpha', 'FDR.nominal']).describe(percentiles=[])
print(results)

fig, ax = plt.subplots(figsize=(12, 6))
sns.pointplot(x="Amplitude", y="Power", hue="Method", data=results)
plt.show()

fig1, ax1 = plt.subplots(figsize=(12, 6))
plt.plot([-1, np.max(signal_amplitude_vec)+1],
         [nominal_fdr, nominal_fdr], linewidth=1, linestyle="--", color="red")
sns.pointplot(x="Amplitude", y="FDP", hue="Method", data=results)
plt.ylim([0, 0.25])
plt.show()
