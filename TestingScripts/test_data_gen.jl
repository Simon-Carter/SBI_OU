using Distributions, LinearAlgebra, MLUtils, Random
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