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
    n = div(length(y_pred) , 2)
    half1 = @view  ypred[1:n]
    half2 = @view ypred[n+1:end]
    negloglike = (0.5)*(((data) .- half1)./half2).^2 + log.(half2.*sqrt(2*pi))
    negloglike = sum(negloglike)
    print(negloglike)
end