using Lux, Optimisers, Random, Zygote

#= Example Loss function, commented out since only used as reference


function loss_function(model, ps, st, data)
    y_pred, st = Lux.apply(model, data[1], ps, st)
    n = length(y_pred) / 2
    half1 = @view  ypred[1:n]
    half2 = @view ypred[n+1:end]
    negloglike = y_pred[1:n] + 
    mse_loss = mean(abs2, y_pred .- data[2])
    return mse_loss, st, ()
end
=#



function mod_loss_function(y_pred, data)
    # print(size(y_pred),size(data))
    n = div(size(y_pred)[1], 2)
    half1 = @view  y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]
    negloglike = (0.5).*(((data) .- half1)./half2).^2 .+ log.(half2.*sqrt(2*pi))
    negloglike = sum(negloglike)
    print(negloglike)
end



# for lux.jl loss function needts to take 4 parameter,  and return 3 parameters

# input: model, parameters, states and data.

# output: loss, updated_state, and any computed statistics

function lux_gaussian_made_loss(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    # print(y_pred)
    loss = mod_loss_function(y_pred, data)
    return loss, st, ()
end






