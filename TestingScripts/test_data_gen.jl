using Distributions, LinearAlgebra

mu = [3,5,9]
sigma = I(3)

true_dist = MvNormal(mu,sigma)
gen_data = rand(true_dist,1000)
