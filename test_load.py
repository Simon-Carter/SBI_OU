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



import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    for t in range(0, N - 1):
        X[:, t + 1] = theta + np.exp(-kappa * dt) * (X[:, t] - theta) + std_dt * W[:, t]

    X_T = X[:, -1]  # values of X at time T
    return X[1, :]





num_dim = 2
prior = utils.BoxUniform(low=0 * torch.ones(num_dim), high=5 * torch.ones(num_dim))

# make a SBI-wrapper on the simulator object for compatibility
simulator_wrapper, prior = prepare_for_sbi(ou_process, prior)

observation = ou_process([1, 1])


neural_posterior = utils.posterior_nn(
    model="maf", embedding_net=embedding_net, hidden_features=10, num_transforms=2
)

inference = SNPE(prior=prior, density_estimator=neural_posterior)

density_estimator = torch.load('./test_save')

posterior = inference.build_posterior(density_estimator)

posterior_samples = posterior.sample((50000,), x=observation)

plt.plot(observation)
plt.savefig("ou_1_1_1.png")
# samples = posterior.sample((10000,), x=observation).numpy()
#log_probability = posterior.log_prob(samples, x=observation)
post_graph = analysis.pairplot(
    posterior_samples,
    points=torch.tensor([1, 1]),
    limits=[[0, 5], [0, 5]], 
    points_colors="r",
    points_offdiag={"markersize": 6},
    figsize=(6, 6)
)
plt.savefig("plt_graph_111_summary_test_load2.png")

# List the methods of the object using dir()
methods = dir(density_estimator)

# Print the list of methods
for method in methods:
    print(method)

torch.save(density_estimator, './test_save')