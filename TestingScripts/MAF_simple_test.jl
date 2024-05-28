using Lux, Optimisers, Random, Zygote, ADTypes, LinearAlgebra, ConcreteStructs, OneHotArrays


include("../loss_function.jl")
include("../TestingScripts/test_data_gen.jl")
include("../Masked_layer.jl")

import MLDatasets: MNIST

import MLUtils: DataLoader, splitobs

#generate random number generator
rng = MersenneTwister()
Random.seed!(rng, 44497)

# set the optimiser model
opt = Adam(0.003)


model1 = conditional_MADE(MaskedLinear(289, 2, relu), MaskedLinear(2, 2), random_order=false)
model2 = conditional_MADE(MaskedLinear(289, 2, relu), MaskedLinear(2, 2), random_order=true)
#model3 = conditional_MADE(MaskedLinear(6, 4, relu), MaskedLinear(4, 6), random_order=true)
#model4 = conditional_MADE(MaskedLinear(5, 3, relu), MaskedLinear(3, 6), random_order=true)
#model5 = conditional_MADE(MaskedLinear(5, 3, relu), MaskedLinear(3, 6), random_order=true)
#model6 = conditional_MADE(MaskedLinear(3, 3, relu), MaskedLinear(3, 4), random_order=true)
#model7 = conditional_MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)


#model = conditional_MAF(model1, model2, model3, model4, model5, model6)

#=
model1 = conditional_MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=false)
model2 = conditional_MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model3 = conditional_MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model4 = conditional_MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model5 = conditional_MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
model6 = conditional_MADE(MaskedLinear(2, 3, relu), MaskedLinear(3, 4), random_order=true)
=#

model = conditional_MAF(model1, model2, conditional_num=288)

tstate = Lux.Training.TrainState(rng, model, opt);


train_dataloader = generate_dataloader(david, david_noise, 10000, 1, 600000)

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

tstate = main(tstate, vjp_rule, train_dataloader, 500)

b = [sample(model, tstate.parameters, tstate.states, conditional=single_ou_noise(10000,randn()*0.05 + 1,randn()*0.05 + 1.5, randn()*0.05)[4:end]) for i in 1:10000]
b = hcat(b...)
a = iterate(train_dataloader)[1]
scatter(b[1,:],b[2,:])
scatter!(a[1,:],a[2,:])



