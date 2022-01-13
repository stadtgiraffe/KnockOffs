import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Deep_Knockoffs.machine import KnockoffMachine
from Deep_Knockoffs.gaussian import GaussianKnockoffs
import data
import diagnostics
import parameters
from data import load_IHDP_data

# Number of features
p = 25

# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
model = "mstudent"
distribution_params = parameters.GetDistributionParams(model, p)

# Number of training examples
data_train, data_test = load_IHDP_data(type_a=True, i=0)

# Number of training examples
n = data_train['x'].shape[0]

print("Generated a training dataset of size: %d x %d." %(data_train['x'].shape))

# Compute the empirical covariance matrix of the training data
SigmaHat = np.cov(data_train['x'], rowvar=False)

# Initialize generator of second-order knockoffs
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(data_train['x'],0), method="sdp")

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print('Average absolute pairwise correlation: %.3f.' %(np.mean(np.abs(corr_g))))

# Load the default hyperparameters for this model
training_params = parameters.GetTrainingHyperParams(model)

# Set the parameters for training deep knockoffs
pars = dict()
# Number of epochs
pars['epochs'] = 100
# Number of iterations over the full data per epoch
pars['epoch_length'] = 100
# Data type, either "continuous" or "binary"
pars['family'] = "continuous"
# Dimensions of the data
pars['p'] = p
# Size of the test set
pars['test_size']  = 0
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
machine.load(checkpoint_name)

# Compute goodness of fit diagnostics on 50 test sets containing 100 observations each
n_exams = 100
n_samples = 1000
exam = diagnostics.KnockoffExam(DataSampler, {'Machine': machine, 'Second-order': second_order})
diagnostics = exam.diagnose(n_samples, n_exams)

# Summarize diagnostics
diagnostics.groupby(['Method', 'Metric', 'Swap']).describe()

# Plot covariance goodness-of-fit statistics
data = diagnostics[(diagnostics.Metric == "Covariance") & (diagnostics.Swap != "self")]
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
plt.show()

# Plot k-nearest neighbors goodness-of-fit statistics
data = diagnostics[(diagnostics.Metric == "KNN") & (diagnostics.Swap != "self")]
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
plt.show()

# Plot MMD goodness-of-fit statistics
data = diagnostics[(diagnostics.Metric == "MMD") & (diagnostics.Swap != "self")]
fig3, ax3 = plt.subplots(figsize=(12, 6))
sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
plt.show()

# Plot energy goodness-of-fit statistics
data = diagnostics[(diagnostics.Metric == "Energy") & (diagnostics.Swap != "self")]
fig4, ax4 = plt.subplots(figsize=(12,6))
sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
plt.show()

# Plot average absolute pairwise correlation between variables and knockoffs
data = diagnostics[(diagnostics.Metric=="Covariance") & (diagnostics.Swap == "self")]
fig5, ax5 = plt.subplots(figsize=(12,6))
sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
plt.show()
