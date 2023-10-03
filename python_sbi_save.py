import numpy as np
import scipy as scp
import scipy.stats as ss
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from mpl_toolkits import mplot3d
from matplotlib import cm
import scipy.special as scsp
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator


import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi import utils
from sbi import analysis
from sbi import inference
from sbi.inference import SNPE, simulate_for_sbi, prepare_for_sbi
from sbi.inference.base import infer

from matplotlib import pyplot as plt

seed = 0
torch.manual_seed(seed)



#hyper parameters
##  how many convolution layers, number of channels, and kernel size
##  Pooling kernel size and stride
##  Fully connected layer out features



##  As an example lets give 1000 point time series
##  With a single convolution layer, 6 kernels kernel size of 10 - lets read up on the justification of the chossen kernel size and the desired architecture, make sure to include padding to avoid cyclical convolution
##  
## Max pooling layer, benifits of max pooling in 1d, pick artbitrary say reduce 1000 timeseries to 100

## Finally use the fully connected layer to condense it down to the desired summary statistics of dimension 2 or 3, 2 if we have enough data, 3 if we do not


class SummaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 2D convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=100)
        # Maxpool layer that reduces time series from large to small
        self.pool = nn.MaxPool1d(kernel_size=10, stride=10)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=2940, out_features=3)
        
    def forward(self, x):
        x = x.view(-1, 1, 5000)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 2940)
        x = F.relu(self.fc(x))
        return x


embedding_net = SummaryNet()



#OU process, without noise we use this directly as our simulator
def ou_process(params):
    
    params = np.asarray(params)
    
    N = 5000  # time steps
    paths = 2  # number of paths
    T = 50
    T_vec, dt = np.linspace(0, T, N, retstep=True)

    kappa = params[0]
    theta = 0
    sigma = params[1]
    std_asy = np.sqrt(sigma**2 / (2 * kappa))  # asymptotic standard deviation

    X0 = 2
    X = np.zeros((paths, N))
    X[:, 0] = X0
    W = ss.norm.rvs(loc=0, scale=1, size=(paths, N - 1))

    # Uncomment for Euler Maruyama
    # for t in range(0,N-1):
    #    X[:,t+1] = X[:,t] + kappa*(theta - X[:,t])*dt + sigma * np.sqrt(dt) * W[:,t]

    std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    for t in range(0, N - 1):
        X[:, t + 1] = theta + np.exp(-kappa * dt) * (X[:, t] - theta) + std_dt * W[:, t]

    X_T = X[:, -1]  # values of X at time T
    return X[1, :]



# Implement Known summary Statistics for ann OU process

def OU_summary(data):
    if len(data) == 0:
        raise ValueError("Input array is empty")

    # Calculate the sample mean
    mean = np.mean(data)

    # Calculate the sample variance
    variance = np.var(data, ddof=1)

    # Calculate the sample covariance matrix
    autocorrelation_at_lag1 = np.corrcoef(data[:-1], data[1:])[0, 1]

    #to deal with shorter samples
    initial_value = data[0]

    return [mean, variance, autocorrelation_at_lag1, initial_value]


#implement new simulator with handpicked summary statistics
def OU_summary_hand(params):
    return OU_summary(ou_process(params))

# Test Stuff to make sure these functions are working
test_summary = OU_summary_hand(ou_process([10,11]))


### SBI implementation/training

#set the simulation number
sim_num = 300000

## Defining the prior
num_dim = 2
prior = utils.BoxUniform(low=0 * torch.ones(num_dim), high=5 * torch.ones(num_dim))


# make a SBI-wrapper on the simulator object for compatibility
simulator_wrapper, prior = prepare_for_sbi(OU_summary_hand, prior)


#generate an observation
observation = OU_summary_hand([1, 1])
observation2 = OU_summary_hand([2,3])
observation3 = OU_summary_hand([4,1])

# For teting purposes, check if our simulator is consistent with expectations
plt.plot(observation)
plt.savefig("ou_1_1_1.png")


'''
#define the arhitecture of the conditional neural density estimato
neural_posterior = utils.posterior_nn(
    model="maf", embedding_net=embedding_net, hidden_features=10, num_transforms=2
)
'''

#define the arhitecture of the conditional neural density estimato
neural_posterior = utils.posterior_nn(
    model="maf", hidden_features=10, num_transforms=2
)

#Create the inference object
inference = SNPE(prior=prior, density_estimator=neural_posterior)


#Sample the parameters from out simulator
a, b = simulate_for_sbi(simulator_wrapper, prior, num_simulations=sim_num)

torch.save((a,b), f'./Models/OU_samples_{sim_num}')

#Give the simulator our parameters
inference = inference.append_simulations(a, b)

#train the neural network
density_estimator = inference.train(stop_after_epochs=100)

#save our model
torch.save(density_estimator, './Models/test_save')
torch.save(embedding_net, './Models/embedding_save')


#generate the posterior from our neural density estimator
posterior = inference.build_posterior(density_estimator)


#sample from our posterior
posterior_samples = posterior.sample((50000,), x=observation)


### Graphing Stuff
post_graph = analysis.pairplot(
    posterior_samples,
    points=torch.tensor([1, 1]),
    limits=[[0, 5], [0, 5]], 
    points_colors="r",
    points_offdiag={"markersize": 6},
    figsize=(6, 6)
)
plt.savefig(f"plt_graph_111_summary_3_nonn_{sim_num}.png")

posterior_samples2 = posterior.sample((50000,), x=observation2)


### Graphing Stuff
post_graph = analysis.pairplot(
    posterior_samples2,
    points=torch.tensor([2, 3]),
    limits=[[0, 5], [0, 5]], 
    points_colors="r",
    points_offdiag={"markersize": 6},
    figsize=(6, 6)
)
plt.savefig(f"plt_graph_23_summary_3_nonn_{sim_num}.png")

posterior_samples3 = posterior.sample((50000,), x=observation3)

### Graphing Stuff
post_graph = analysis.pairplot(
    posterior_samples3,
    points=torch.tensor([4, 1]),
    limits=[[0, 5], [0, 5]], 
    points_colors="r",
    points_offdiag={"markersize": 6},
    figsize=(6, 6)
)
plt.savefig(f"plt_graph_41_summary_3_nonn_{sim_num}.png")

''' Debugging Code
# List the methods of the object using dir()
methods = dir(density_estimator)

# Print the list of methods
for method in methods:
    print(method)
'''