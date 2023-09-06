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
        self.fc = nn.Linear(in_features=6 * 100, out_features=2)
        
    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6 * 4 * 4)
        x = F.relu(self.fc(x))
        return x


embedding_net = SummaryNet()



def ou_process(params):
    
    params = np.asarray(params)
    
    Θ = params[0]
    μ = params[1]
    σ = params[2]
    
    dt = .001  # Time step.
    T = 300.  # Total time.
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.
    
    sigma_bis = σ * np.sqrt(2*Θ)
    sqrtdt = np.sqrt(dt)
    
    x = np.zeros(n)
    
    for i in range(n - 1):
        x[i + 1] = x[i] + dt * Θ * (-(x[i] - μ)) + sigma_bis * sqrtdt * np.random.randn()
    
    return x




prior = utils.BoxUniform(
    low=torch.tensor([0.0, -5.0, 0.0]), high=torch.tensor([5.0, 5.0, 5.0])
)



posterior = infer(
    ou_process, prior, method="SNPE", num_simulations=1000, num_workers=4,
)

observation = ou_process([1, 0, 1])

plt.plot(observation)
plt.savefig("ou_1_0_1.png")


samples = posterior.sample((1000,), x=observation).numpy()
log_probability = posterior.log_prob(samples, x=observation)
post_graph = analysis.pairplot(samples, limits=[[-5, 5], [-5, 5], [-5, 5]], figsize=(6, 6))
plt.savefig("plt_graph.png")

print(np.mean(samples, axis=0))