using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays


include("loss_function.jl")
include("./TestingScripts/test_data_gen.jl")
include("./Masked_layer.jl")

import MLDatasets: MNIST

import MLUtils: DataLoader, splitobs

#generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 12345)

# set the optimiser model
opt = Adam(0.060)

model = MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4))
model2 = MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model3 = MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model = MAF(model, model2, model3)

tstate = Lux.Training.TrainState(rng, model, opt);

train_dataloader = generate_data_maf_paper(40000,5000)

vjp_rule = Lux.Training.AutoZygote()
ADTypes.AutoZygote()

function main(tstate::Lux.Experimental.TrainState, vjp, data_loader, epochs)
    for epoch in 1:epochs
        for data in data_loader
            grads, loss, stats, tstate = Lux.Training.compute_gradients(vjp,
            lux_gaussian_maf_loss, data, tstate)
            println("Epoch: $(epoch) || Loss: $(loss)")
            tstate = Lux.Training.apply_gradients(tstate, grads)
        end
    end
    return tstate
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstate = main(tstate, vjp_rule, train_dataloader, 1000)