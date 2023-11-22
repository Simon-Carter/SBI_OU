using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs 

include("loss_function.jl")
include("./TestingScripts/test_data_gen.jl")
include("./Masked_layer.jl")

#generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 12345)

# set the optimiser model
opt = Adam(0.03f0)

# st the architecture of the neural network
model = Chain(Dense(3 => 4, relu), MaskedLinear(4 => 5), relu, Dense(5 => 6))

tstate = Lux.Training.TrainState(rng, model, opt);

gen_data = gen_data_fun([1,2,3], 2.718*I(3), 1000, 1000)

vjp_rule = Lux.Training.AutoZygote()
ADTypes.AutoZygote()

function main(tstate::Lux.Experimental.TrainState, vjp, data_loader, epochs)
    for epoch in 1:epochs
        for data in data_loader
            grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
            lux_gaussian_made_loss, data, tstate)
            println("Epoch: $(epoch) || Loss: $(loss)")
            tstate = Lux.Training.apply_gradients(tstate, grads)
        end
    end
    return tstate
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()



tstate = main(tstate, vjp_rule, gen_data, 50000)
y_pred = dev_cpu(Lux.apply(tstate.model, gen_data.data, tstate.parameters, tstate.states)[1])
# test loss compared to true values


#y_true = repeat([3;5;9;0;0;0], 1, 100000)
#test_loss = log_std_loss(y_true, gen_data)

#=

y_true2 = repeat([2;5;9;-15;-10;-6], 1, 1000)
test_loss2 = log_std_loss(y_true2, gen_data)

print(test_loss,"   ",test_loss2)
=#

