using Lux, Optimisers, Random, Zygote

function log_std_loss(y_pred, data)
    print(size(y_pred),size(data))
    #print(data)
    n = div(size(y_pred)[1], 2)
    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]
    #println(n, half1, half2)
    u = (data.-half1).*exp.(-half2)
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2) .+ half2
    negloglike = mean(negloglike, dims=2)
    negloglike = sum(negloglike)
    if (negloglike == Inf) 
        DomainError(val) 
    end
    return negloglike
end


# for lux.jl loss function needts to take 4 parameter,  and return 3 parameters

# input: model, parameters, states and data.

# output: loss, updated_state, and any computed statistics

function lux_gaussian_made_loss(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    loss = log_std_loss(y_pred, data)
    return loss, st, ()
end






