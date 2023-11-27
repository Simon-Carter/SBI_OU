test_data = [1 3 2;1 4 7;2 2 2]

y_pred = [1 3 3;1 2 7;2 2 10;2 2 2; 2 2 2; 5 5 5]

include("../loss_function.jl")
# st the architecture of the neural network
model = Chain(Dense(3 => 4, relu), Dense(4 => 5, relu), Dense(5 => 6))

print(mod_loss_function(y_pred,test_data))

print(log_std_loss(y_pred,test_data))