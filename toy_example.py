import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import data
import diagnostics
import parameters
from data import load_IHDP_data

data_train, data_test = load_IHDP_data(type_a=True, i=0)
# Number of features
p = 25

# Load the built-in Gaussian model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
model = "gaussian"
distribution_params = parameters.GetDistributionParams(model, p)

# Initialize the data generator
DataSampler = data.DataSampler(distribution_params)

# Number of training examples
n = data_train['x'].shape[0]

# Sample training data
# data_train['x'] = DataSampler.sample(n)
print("Generated a training dataset of size: %d x %d." %(data_train['x'].shape))

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
pars['epochs'] = 200
# Number of iterations over the full data per epoch
pars['epoch_length'] = 50
# Data type, either "continuous" or "binary"
pars['family'] = "continuous"
# Dimensions of the data
pars['p'] = p
# Size of the test set
pars['test_size'] = int(0.1*n)
# Batch size
pars['batch_size'] = int(0.45*n)
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

# Initialize the machine
machine = KnockoffMachine(pars)

# Train the machine
print("Fitting the knockoff machine...")
machine.train(data_train['x'])

# Generate deep knockoffs
Xk_train_m = machine.generate(data_train['x'])
print("Size of the deep knockoff dataset: %d x %d." %(Xk_train_m.shape))

# Generate second-order knockoffs
Xk_train_g = second_order.generate(data_train['x'])
print("Size of the second-order knockoff dataset: %d x %d." %(Xk_train_g.shape))

# Plot diagnostics for deep knockoffs
diagnostics.ScatterCovariance(data_train['x'], Xk_train_m)

# Sample test data
# data_test['x'] = DataSampler.sample(n, test=True)
print("Generated a test dataset of size: %d x %d." %(data_test['x'].shape))

# Generate deep knockoffs
Xk_test_m = machine.generate(data_test['x'])
print("Size of the deep knockoff test dataset: %d x %d." %(Xk_test_m.shape))

# Generate second-order knockoffs
Xk_test_g = second_order.generate(data_test['x'])
print("Size of the second-order knockoff test dataset: %d x %d." %(Xk_test_g.shape))

# # Generate oracle knockoffs
# oracle = GaussianKnockoffs(DataSampler.Sigma, method="sdp", mu=DataSampler.mu)
# Xk_test_o = oracle.generate(data_test['x'])
# print("Size of the oracle knockoff test dataset: %d x %d." %(Xk_test_o.shape))

# Plot diagnostics for deep knockoffs
diagnostics.ScatterCovariance(data_test['x'], Xk_test_m)

# Plot diagnostics for second-order knockoffs
diagnostics.ScatterCovariance(data_test['x'], Xk_test_g)

# # Plot diagnostics for oracle knockoffs
# diagnostics.ScatterCovariance(data_test['x'], Xk_test_o)


# Compute goodness of fit diagnostics on 50 test sets containing 100 observations each
n_exams = 50
n_samples = 100
exam = diagnostics.KnockoffExam(DataSampler, {'Machine':machine, 'Second-order':second_order})
diagnostics = exam.diagnose(n_samples, n_exams)

# Summarize diagnostics
diagnostics.groupby(['Method', 'Metric', 'Swap']).describe()

# Plot covariance goodness-of-fit statistics
data = diagnostics[(diagnostics.Metric == "Covariance") & (diagnostics.Swap != "self")]
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
plt.show()

# Plot k-nearest neighbors goodness-of-fit statistics
data = diagnostics[(diagnostics.Metric == "KNN") & (diagnostics.Swap != "self")]
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
plt.show()
