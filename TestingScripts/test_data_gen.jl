using Distributions, LinearAlgebra

function gen_data_fun(mu, sigma, n)
    true_dist = MvNormal(mu,sigma)
    gen_data = rand(true_dist,n)
    return gen_data

end