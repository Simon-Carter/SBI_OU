
"""
    Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform,
          init_bias=zeros32, bias::Bool=true)

Create a traditional fully connected layer, whose forward pass is given by:
`y = activation.(weight * x .+ bias)`

## Arguments

  - `in_dims`: number of input dimensions
  - `out_dims`: number of output dimensions
  - `activation`: activation function

## Keyword Arguments

  - `init_weight`: initializer for the weight matrix
    (`weight = init_weight(rng, out_dims, in_dims)`)
  - `init_bias`: initializer for the bias vector (ignored if `use_bias=false`)
  - `use_bias`: Trainable bias can be disabled entirely by setting this to `false`
  - `allow_fast_activation`: If `true`, then certain activations can be approximated with
    a faster version. The new activation function will be given by
    `NNlib.fast_act(activation)`

## Input

  - `x` must be an AbstractArray with `size(x, 1) == in_dims`

## Returns

- AbstractArray with dimensions `(out_dims, ...)` where `...` are the dimensions of `x`
- Empty `NamedTuple()`

## Parameters

- `weight`: Weight Matrix of size `(out_dims, in_dims)`
- `bias`: Bias of size `(out_dims, 1)` (present if `use_bias=true`)
"""
@concrete struct MaskedLinear <: Lux.AbstractExplicitLayer
  activation
  in_dims::Int
  out_dims::Int
  init_weight
  init_bias
  init_mask::Base.RefValue{Matrix{Float32}}
end

function Base.show(io::IO, d::MaskedLinear)
  print(io, "MaskedLinear($(d.in_dims) => $(d.out_dims)")
  return print(io, ")")
end

function MaskedLinear(mapping::Pair{<:Int, <:Int}; kwargs...)
  print("Masked_linear construcot called")
  println(kwargs...)
  return MaskedLinear(first(mapping), last(mapping); kwargs...)
end

# added a mak to the constructor
#used a reference since it needs to be mutable, and that cant happen with direct storage in a concrete structure
function MaskedLinear(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform,
        init_bias=zeros32)
        init_mask=ones(Float32, out_dims, in_dims)
        init_mask_ref = Ref(init_mask)
  return MaskedLinear(activation, in_dims, out_dims, init_weight, init_bias, init_mask_ref)
end

function Lux.initialparameters(rng::AbstractRNG, d::MaskedLinear)
    return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
        bias=d.init_bias(rng, d.out_dims, 1))
end

function Lux.parameterlength(d::MaskedLinear)
    return d.out_dims * (d.in_dims + 1)
end

# good for efficiency not exactly sure why yet
Lux.statelength(d::MaskedLinear) = 0


# modified standard dense layer to implement the mask value pointed to by the pointer
@inline function (d::MaskedLinear)(x::AbstractVecOrMat, ps, st::NamedTuple)
    return d.activation.(((d.init_mask[]).*ps.weight)*x .+ ps.bias), st
end


# -------------------------------------------------------------------
# MADE Container Layer


# TODO GIve a more detailed comment on this layer onsistent with the others
# MADE container - containter of MaskedLinear Layers (implemented as a Guassian Made)
struct MADE{T <: NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
  layers::T
  mask::Base.RefValue{}
end

# Implements the sample function to sample from the distribution represented by the MADE container
function sample(T::MADE, ps, st; samples = randn(T.layers[1].in_dims))
  input = T.layers[1].in_dims
  println(samples)
  for i in 1:input
    mean = T(samples, ps, st)[1][i]
    std = exp(T(samples, ps, st)[1][i+input])
    samples[i] = std*samples[i] + mean
  end
  return samples
end


#Generates a seet of integers for a layer consistent with the autoregressive property
#used to calculate the mask
function generate_m_k(layers)
  dims = [(i.in_dims, i.out_dims) for i in layers]

  D = dims[1][1]

  integer_assign = []
  push!(integer_assign,1:D)
  for i in dims[1:end-1]
    push!(integer_assign, rand(1:D-1, i[2]))
  end
  push!(integer_assign,1:D)

  return(integer_assign)
end


#Calculate masks for each layer and pushes them to an array in order to be sent to the layer
#gaussianMADE only one implemented 
function generate_masks(m_k, gaussianMADE::Bool)
  Masks = []
  for i in eachindex(m_k[1:end-2]) 
    pair = collect(Iterators.product(m_k[i], m_k[i+1]))
    M = zeros(size(pair))
    foreach(i -> pair[i][1] > pair[i][2] ?  M[i] = 0 : M[i] = 1, CartesianIndices(pair))
    push!(Masks, M')
  end

  pair = collect(Iterators.product(m_k[end-1], m_k[end]))
  M = zeros(size(pair))
  foreach(i -> pair[i][1] >= pair[i][2] ?  M[i] = 0 : M[i] = 1, CartesianIndices(pair))

  if gaussianMADE
    M = hcat(M,M)
  end

  push!(Masks, M')

  return(Masks)
end


#Constructor for the MADE layer
# sets initial mask
function MADE(layers...; gaussianMADE::Bool=true)
  names = ntuple(i -> Symbol("layer_$i"), length(layers))
  mask = generate_masks(generate_m_k(layers), gaussianMADE)
  mask_ref = Ref(mask)

  for i in eachindex(layers)
    set_mask(layers[i],mask[i])
  end

  return MADE(NamedTuple{names}(layers), mask_ref)
end

# Just Sets Mask for a particular layer
function set_mask(layer::MaskedLinear, mask)
  layer.init_mask[] = mask
end

(c::MADE)(x, ps, st::NamedTuple) = applyMADE(c.layers, x, ps, st, c.mask)

#TODO Figure out why these generated funtions are used, probably for optimization reasons
# essentially just the forward pass
@generated function applyMADE(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}, masks) where {fields}
N = length(fields)
x_symbols = vcat([:x], [gensym() for _ in 1:N])
st_symbols = [gensym() for _ in 1:N]

calls = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]

push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st))
return Expr(:block, calls...)
end


MADE(; kwargs...) = MADE((; kwargs...))


#-------------------------------------------------------------------------------------------------------------
# MAF layer (chain of MADE)


struct MAF{T <: NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
  layers::T
end

function MAF(layers...;)
  names = ntuple(i -> Symbol("MADE_$i"), length(layers))
  return MAF(NamedTuple{names}(layers))
end

(c::MAF)(x, ps, st::NamedTuple) = applyMAF(c.layers, x, ps, st)

# simple macro that transforms x to there correspoding random variable representation
#used in the flow part of Masked autoregressive flow
@inline function coord_transform(x, y_pred)
    n = div(size(y_pred)[1], 2)
    half1 = @view y_pred[1:n,:]
    half2 = @view y_pred[n+1:end,:]
  return (x .- half1).*exp.(-half2)
end

# forward pass, use the coord transform
# TODO Test this and make sure its not causing the bug that keeps coming up
@generated function applyMAF(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}) where {fields}
N = length(fields)
x_symbols = vcat([:x], [gensym() for _ in 1:N])
st_symbols = [gensym() for _ in 1:N]
calls1 = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]
calls2 = [:($(x_symbols[i]) = coord_transform($(x_symbols[i-1]),$(x_symbols[i]))) for i in 2:N]


n = length(calls1) + length(calls2)
calls = similar(calls1, n)

calls[1:2:n] .= calls1
calls[2:2:n] .= calls2

push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st))
return Expr(:block, calls...)
end


#The sample function for the MAF
#TODO Also needs to verify this is not causing the bug
function sample(T::MAF, ps, st)
  _sample = randn(T.layers[1].layers[1].in_dims)
  for i in reverse(eachindex(T.layers))
    _sample = sample(T.layers[i], ps[i], st[i], samples = _sample)
  end
  return _sample
end