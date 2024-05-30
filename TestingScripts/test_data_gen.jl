using Distributions, LinearAlgebra, MLUtils, Random, DifferentialEquations, StatsBase
using NMoons, Plots



function gen_data_fun(mu, sigma, n, batch_size)
    true_dist = MvNormal(mu,sigma)
    gen_data = rand(true_dist,n)
    loader = DataLoader(gen_data; batchsize=batch_size, shuffle=true)
    return loader

end

# test stuff

t = gen_data_fun([1,2,3], I(3), 100, 10)
for i in t
    println(i)
    println("")
end

function generate_data_made_2class(n,batch_size)
    class1 = [1,1,0,0,0,0]
    class1_matrix = repeat(class1, 1, div(n,2))
    class2 = [0,0,0,0,1,1]
    class2_matrix = repeat(class2, 1, div(n,2))
    data= hcat(class1_matrix,class2_matrix)
    permutation = randperm(n)
    data = (0.1*(rand(size(data)...))).+ data
    loader = DataLoader(data[:, permutation]; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_maf_paper(n,batch_size)
    class1 = randn(n)*2
    class2 = randn(n) .+ (1/4).*(class1.^2)

    data = vcat(class1',class2')
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_made_simple(n,batch_size)
    class1 = randn(n)*2
    class2 = randn(n) .+ (1/4).*(class1)

    data = vcat(class1',class2')
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_moons(n,batch_size)
    X, _ = nmoons(Float64, n, 3, ε=0.3, d=2, repulse=(-0.25, -0.25))
    loader = DataLoader(X; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_moons_2(n,batch_size)
    X, _ = nmoons(Float64, n, 2, ε=0.05, d=2, repulse=(-0.25, -0.25))
    loader = DataLoader(X; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_moons_2(n,batch_size)
    X, _ = nmoons(Float64, n, 2, ε=0.05, d=2, repulse=(-0.25, -0.25))
    loader = DataLoader(X; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_moons_2_swap(n,batch_size)
    X, _ = nmoons(Float64, n, 2, ε=0.05, d=2, repulse=(-0.25, -0.25))
    X[1, :], X[2, :] = X[2, :], X[1, :]
    loader = DataLoader(X; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_bimodal(n,batch_size)
    X = reshape(randn(2*n), 2, :)
    X[:, 1:Int((n/2))] = X[:, 1:Int((n/2))] .- 5
    X[:, Int((n/2 + 1)):end] = X[:, Int((n/2 +1)):end] .+ 5
    loader = DataLoader(X; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_bimodal_y(n,batch_size)
    X = reshape(randn(2*n), 2, :)
    X[2, 1:Int((n/2))] = X[2, 1:Int((n/2))] .- 5
    X[2, Int((n/2 + 1)):end] = X[2, Int((n/2 +1)):end] .+ 5
    loader = DataLoader(X; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_data_moons_2_fake_conditional(n,batch_size)
    X, _ = nmoons(Float64, n, 2, ε=0.05, d=2, repulse=(-0.25, -0.25))
    X_conditional = [X; randn(1, size(X, 2))]
    loader = DataLoader(X_conditional; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_polynomial(n, batch_size)
    x = (rand(n) .- 0.5).*4
    y = evalpoly.(x, ((0, -2, 1),)) .+ randn(rng, (1, )) .* 0.1f0
    data = vcat(x',y')
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

function single_ou(n, θ, σ)
    μ = 0.0

    OU = OrnsteinUhlenbeckProcess(θ, μ, σ, 0.0, 0.0)

    prob = NoiseProblem(OU, (0.0, (n-1)/10.0 - 0.00000000001))
    sol = solve(prob; dt = 0.1)

    summary = ou_sumary_statistics_calculator(sol.u)

    conditional= [θ; σ; summary]

    return conditional

end

function single_ou_noise(n, θ, σ, noise)
    μ = 0.0

    OU = OrnsteinUhlenbeckProcess(θ, μ, σ, 0.0, 0.0)

    prob = NoiseProblem(OU, (0.0, (n-1)/10.0 - 0.00000000001))
    sol = solve(prob; dt = 0.1)
    sol.u .+= (randn.(size(sol.u))[1].*noise)


    summary = ou_sumary_statistics_calculator(sol.u)

    conditional= [θ; σ; noise; summary]

    return conditional

end

function single_david(data)
end

function ou_sumary_statistics_calculator(data)
    dmean = mean(data)
    dvariance = var(data)
    dautocor = autocor(data, [1])
    return [dmean, dvariance, dautocor...]
end

function generate_ou(n, batch_size)
    data = Array{Float64}(undef, 5, n)
    for i in 1:n
        data[:,i] = single_ou(10000,randn()*0.05 + 1,randn()*0.05 + 1.5)
    end
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_ou_noise(n, batch_size)
    data = Array{Float64}(undef, 5, n)
    for i in 1:n
        data[:,i] = single_ou_noise(10000,randn()*0.05 + 1,randn()*0.05 + 1.5, 0.05)
    end
    loader = DataLoader(data; batchsize=batch_size, shuffle=true)
    return loader
end

function generate_dataloader(BlackBox, prior, batchsize, output_dimension, output_size)
    # generate the undefined array
    data_complete  = Array{Float64}(undef, output_dimension, output_size)

    #multithreaded for loop
    Threads.@threads for i = 1:output_size
        data_complete[:,i]= BlackBox(prior()...)
    end

    loader = DataLoader(data_complete; batchsize=batchsize, shuffle=true)
    return loader
end

function ou_prior()
    return randn()*0.05 + 1,randn()*0.05 + 1.5
end

function ou(θ, σ)
    return single_ou(1000, θ, σ)
end

function ou_noise(θ, σ, noise)
    return single_ou_noise(1000, θ, σ, noise)
end

function ou_noise_prior()
    return randn()*0.05 + 1,randn()*1 + 1, randn()*0.05
end
