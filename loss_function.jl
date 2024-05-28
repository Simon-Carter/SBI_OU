using Lux, Optimisers, Random, Zygote

function log_std_loss(y_pred, data)
    print(size(y_pred),size(data))
    #print(data)
    n = div(size(y_pred)[1], 2)
    data = data[1:n]


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

function log_std_loss2(y_pred, data, extra)
    sum_output = sum(extra)
    println(size(y_pred),size(data))
    #print(data)
    n = div(size(y_pred)[1], 2)
    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]

    half2_all = @view sum_output[n+1:end,:]
    #println(n, half1, half2)
    u = (data.-half1).*exp.(-half2)
    println("This is right before I need it")
    println(size(u), size(half2_all))
    negloglike = 0.5*log(2*pi) .+ 0.5.*(u.^2) .+ half2_all
    negloglike = mean(negloglike, dims=2)
    negloglike = sum(negloglike)
    if (negloglike == Inf) 
        DomainError(val) 
    end
    println("about to return negloklike")
    return negloglike
end

function log_MAF_loss(u)
    n = length(u)
    mu = (1/n)*sum(u)
    sigma = (1/n)*sum((mu .- u).^2)

    return (mu^2 + (sigma -1)^2)
end


# for lux.jl loss function needts to take 4 parameter,  and return 3 parameters

# input: model, parameters, states and data.

# output: loss, updated_state, and any computed statistics

function lux_gaussian_made_loss(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    loss = log_std_loss(y_pred, data)
    return loss, st, ()
end

function lux_gaussian_maf_loss(model, ps, st, data)
    println("loss function called")
    y, st, x1, x2...  = Lux.apply(model, data, ps, st)
    #println(x1)
    #println(size(x2))
    loss = log_std_loss2(y, x1, x2) #TODO double check this
    return loss, st, ()
end





