# Model for Simon
using ModelingToolkit
using StochasticDiffEq
using Plots
using MAT
using LinearAlgebra

"""
Wrapper function around spectral dcm model. FOr simplicity we a re currently only using
    one 
"""


function david_dcm(lnκ_value)

    """
    Maximum likelihood estimator of a multivariate, or vector auto-regressive model.
        y : MxN Data matrix where M is number of samples and N is number of dimensions
        p : time lag parameter, also called order of MAR model
        return values
        mar["A"] : model parameters is a NxNxP tensor, i.e. one NxN parameter matrix for each time bin k ∈ {1,...,p}
        mar["Σ"] : noise covariance matrix
    """
    function mar_ml(y, p)
        (ns, nd) = size(y)
        ns < nd && error("error: there are more covariates than observation")
        y = transpose(y)
        Y = y[:, p+1:ns]
        X = zeros(nd*p, ns-p)
        for i = p:-1:1
            X[(p-i)*nd+1:(p-i+1)*nd, :] = y[:, i:ns+i-p-1]
        end

        A = (Y*X')/(X*X')
        ϵ = Y - A*X
        Σ = ϵ*ϵ'/ns   # unbiased estimator requires the following denominator (ns-p-p*nd-1), the current is consistent with SPM12
        A = -[A[:, (i-1)*nd+1:i*nd] for i = 1:p]    # flip sign to be consistent with SPM12 convention
        mar = Dict([("A", A), ("Σ", Σ), ("p", p)])
        return mar
    end

    """
    This function converts multivariate auto-regression (MAR) model parameters to a cross-spectral density (CSD).
    A     : coefficients of MAR model, array of length p, each element contains the regression coefficients for that particular time-lag.
    Σ     : noise covariance matrix of MAR
    p     : number of time lags
    freqs : frequencies at which to evaluate the CSD
    sf    : sampling frequency

    This function returns:
    csd   : cross-spectral density matrix of size MxN; M: number of samples, N: number of cross-spectral dimensions (number of variables squared)
    """
    function mar2csd(mar, freqs, sf)
        Σ = mar["Σ"]
        p = mar["p"]
        A = mar["A"]
        nd = size(Σ, 1)
        w  = 2*pi*freqs/sf
        nf = length(w)
        csd = zeros(ComplexF64, nf, nd, nd)
        for i = 1:nf
            af_tmp = I
            for k = 1:p
                af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
            end
            iaf_tmp = inv(af_tmp)
            csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'
        end
        csd = 2*csd/sf
        return csd
    end


    vars = matread("./matlab_simulation.mat")

    @parameters t
    D = Differential(t)
    sts = @variables x(t)[1:3] s(t)[1:3] lnu(t)[1:3] lnν(t)[1:3] lnq(t)[1:3] bold(t)[1:3] [irreducible=true]

    @parameters A[1:9] = vec(vars["A"])
    @parameters lnτ[1:3] = vec(vars["transit"])
    @parameters lnκ = lnκ_value

    @brownian η

    H = [0.64, 0.32, 2.00, 0.32, 0.4]
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE
    lnϵ = 0

    eqs = [
            D(x[1])   ~ A[1]*x[1] + A[2]*x[2] + A[3]*x[3] + 0.1η,
            D(x[2])   ~ A[4]*x[1] + A[5]*x[2] + A[6]*x[3] + 0.1η,
            D(x[3])   ~ A[7]*x[1] + A[8]*x[2] + A[9]*x[3] + 0.1η,
            D(s[1])   ~ 0.1x[1] - H[1]*exp(lnκ)*s[1] - H[2]*(exp(lnu[1]) - 1),
            D(s[2])   ~ 0.1x[2] - H[1]*exp(lnκ)*s[2] - H[2]*(exp(lnu[2]) - 1),
            D(s[3])   ~ 0.1x[3] - H[1]*exp(lnκ)*s[3] - H[2]*(exp(lnu[3]) - 1),
            D(lnu[1]) ~ s[1] / exp(lnu[1]),
            D(lnu[2]) ~ s[2] / exp(lnu[2]),
            D(lnu[3]) ~ s[3] / exp(lnu[3]),
            D(lnν[1]) ~ (exp(lnu[1]) - exp(lnν[1])^(H[4]^-1)) / (H[3]*exp(lnτ[1])*exp(lnν[1])),
            D(lnν[2]) ~ (exp(lnu[2]) - exp(lnν[2])^(H[4]^-1)) / (H[3]*exp(lnτ[2])*exp(lnν[2])),
            D(lnν[3]) ~ (exp(lnu[3]) - exp(lnν[3])^(H[4]^-1)) / (H[3]*exp(lnτ[3])*exp(lnν[3])),
            D(lnq[1]) ~ (exp(lnu[1])/exp(lnq[1])*((1 - (1 - H[5])^(exp(lnu[1])^-1))/H[5]) - exp(lnν[1])^(H[4]^-1 - 1))/(H[3]*exp(lnτ[1])),
            D(lnq[2]) ~ (exp(lnu[2])/exp(lnq[2])*((1 - (1 - H[5])^(exp(lnu[2])^-1))/H[5]) - exp(lnν[2])^(H[4]^-1 - 1))/(H[3]*exp(lnτ[2])),
            D(lnq[3]) ~ (exp(lnu[3])/exp(lnq[3])*((1 - (1 - H[5])^(exp(lnu[3])^-1))/H[5]) - exp(lnν[3])^(H[4]^-1 - 1))/(H[3]*exp(lnτ[3])),
            # bold[1] ~ V0*(k1 - k1*exp(lnq[1]) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(lnq[1])/exp(lnν[1]) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν[1])),
            # bold[2] ~ V0*(k1 - k1*exp(lnq[2]) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(lnq[2])/exp(lnν[2]) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν[2])),
            # bold[3] ~ V0*(k1 - k1*exp(lnq[3]) + exp(lnϵ)*r0*E0*TE - exp(lnϵ)*r0*E0*TE*exp(lnq[3])/exp(lnν[3]) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν[3]))
            ]

    @mtkbuild sys = System(eqs, t)

    x0 = zeros(length(unknowns(sys)))
    sf = 10;
    prob = SDEProblem(sys, x0, (0.0, 200.0))
    sol = solve(prob, EM(), dt=sf^-1)

    plot(sol)

    bold1 = V0*(k1 .- k1*exp.(sol[13, :]) .+ exp(lnϵ)*r0*E0*TE .- exp(lnϵ)*r0*E0*TE*exp.(sol[13, :])./exp.(sol[10, :]) .+ 1 .- exp(lnϵ) .- (1-exp(lnϵ))*exp.(sol[10, :]))
    bold2 = V0*(k1 .- k1*exp.(sol[14, :]) .+ exp(lnϵ)*r0*E0*TE .- exp(lnϵ)*r0*E0*TE*exp.(sol[14, :])./exp.(sol[11, :]) .+ 1 .- exp(lnϵ) .- (1-exp(lnϵ))*exp.(sol[11, :]))
    bold3 = V0*(k1 .- k1*exp.(sol[15, :]) .+ exp(lnϵ)*r0*E0*TE .- exp(lnϵ)*r0*E0*TE*exp.(sol[15, :])./exp.(sol[12, :]) .+ 1 .- exp(lnϵ) .- (1-exp(lnϵ))*exp.(sol[12, :]))

    # final time series
    bold = hcat(bold1, bold2, bold3)

    # compute summary Statistics
    mar = mar_ml(bold, 8)
    freqs = range(128^-1, 8^-1, 32)
    csd = mar2csd(mar, freqs, sf)    # cross spectral density

    #format the output to be compatable with sbi


    #flatten the matrices to single vector
    flat_output = vec(csd)

    #just use real values for now

    return flat_real = real.(flat_output)

    #flat_imag = imag.(flat_output) 
end