using Lux, Optimisers, Random, Zygote

include("loss_function.jl")

#generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 12345)

# set the optimiser model
opt = Adam(0.03f0)

# st the architecture of the neural network
model = Chain(Dense(3 => 4, relu), Dense(4 => 6))

tstate = Lux.Training.TrainState(rng, model, opt);

