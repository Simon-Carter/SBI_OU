using Random


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
  order::AbstractArray{Int}
end

# Implements the sample function to sample from the distribution represented by the MADE container
function sample(T::MADE, ps, st; samples = randn(T.layers[1].in_dims))
  input = T.layers[1].in_dims
  order = sortperm(T.order) # gets the index for the m_k values in increasing order
  println(samples)
  for i in order
    mean = T(samples, ps, st)[1][i]
    std = exp(T(samples, ps, st)[1][i+input])
    samples[i] = std*samples[i] + mean
  end
  return samples
end


#Generates a seet of integers for a layer consistent with the autoregressive property
#used to calculate the mask
function generate_m_k(layers, random_order::Bool; num_conditional=0)
  dims = [(i.in_dims, i.out_dims) for i in layers]

  D = dims[1][1]
  D = D - num_conditional

  integer_assign = []
  if random_order
    push!(integer_assign, randperm(D))
  else
    push!(integer_assign,1:D)
  end

  for i in dims[1:end-1]
    push!(integer_assign, rand(1:D-1, i[2])) #TODO double check the integer assign is working
  end
  push!(integer_assign,integer_assign[1])

  integer_assign[1] = vcat(integer_assign[1], ones(num_conditional))

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
function MADE(layers...; gaussianMADE::Bool=true, random_order::Bool=false)
  names = ntuple(i -> Symbol("layer_$i"), length(layers))

  m_k = generate_m_k(layers, random_order)

  order = m_k[1]

  mask = generate_masks(m_k, gaussianMADE)
  mask_ref = Ref(mask)

  for i in eachindex(layers)
    set_mask(layers[i],mask[i])
  end

  return MADE(NamedTuple{names}(layers), mask_ref, order)
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

# MADE conditional Container Layer


# TODO GIve a more detailed comment on this layer onsistent with the others
# MADE container - containter of MaskedLinear Layers (implemented as a Guassian Made)

struct conditional_MADE{T <: NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
  layers::T
  mask::Base.RefValue{}
  order::AbstractArray{Int}
  num_conditional::Int
end

function sample(T::conditional_MADE, ps, st; samples = randn(T.layers[1].in_dims))
  input = T.layers[1].in_dims
  order = sortperm(T.order) # gets the index for the m_k values in increasing order
  for i in order
    mean = T(samples, ps, st)[1][i]
    std = exp(T(samples, ps, st)[1][i+input])
    samples[i] = std*samples[i] + mean
  end
  return samples
end


#Constructor for the MADE layer
# sets initial mask
function conditional_MADE(layers...; gaussianMADE::Bool=true, random_order::Bool=false)
  names = ntuple(i -> Symbol("layer_$i"), length(layers))

  input_size = layers[1].in_dims
  output_size = layers[end].out_dims

  num_conditional = int(input_size - (output_size / 2))

  m_k = generate_m_k(layers, random_order, num_conditional=num_conditional)

  order = m_k[1]

  mask = generate_masks(m_k, gaussianMADE)
  mask_ref = Ref(mask)

  for i in eachindex(layers)
    set_mask(layers[i],mask[i])
  end

  return conditional_MADE(NamedTuple{names}(layers), mask_ref, order, num_conditional)
end

(c::conditional_MADE)(x, ps, st::NamedTuple) = apply_conditionalMADE(c.layers, x, ps, st, c.mask)

#TODO Figure out why these generated funtions are used, probably for optimization reasons
# essentially just the forward pass
@generated function apply_conditionalMADE(layers::NamedTuple{fields}, x, ps,
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


conditional_MADE(; kwargs...) = conditional_MADE((; kwargs...))

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
    println(x[:,1])
    println(half1[:,1], half2[:,1], y_pred[:,1])
    u = (x .- half1).*exp.(-half2)
    println(u[:,1])
  return u
end

# forward pass, use the coord transform
# TODO Test this and make sure its not causing the bug that keeps coming up
@generated function applyMAF(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}) where {fields}
N = length(fields) #number of MADE layers
x_symbols = vcat([:x], [gensym() for _ in 1:N])
total_std = [gensym() for _ in 1:N]
st_symbols = [gensym() for _ in 1:N]
calls1 = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]
calls2 = [:($(x_symbols[i]) = coord_transform($(x_symbols[i-1]),$(x_symbols[i]))) for i in 2:N]
calls3 = [:($(total_std[i]) = copy($(x_symbols[i+1]))) for i in 1:N]



n = length(calls1) + length(calls2) + length(calls3)
calls = similar(calls1, n)

#add up all the log std for each layer
#each_layer_ouptut = :([$(x_symbols[N]) for i in 1:N])

################# add the definition of total_std as an array in the list of blocks
#calls[1] .= :($(x_symbols[1]) = 1)
calls[1:3:n] .= calls1
calls[2:3:n] .= calls3
calls[3:3:n] .= calls2


push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st, $(x_symbols[N]), $(total_std...)))
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

#-------------------------------------------------------------------------------------------------------------
# conditional MAF layer (chain of MADE with the conditional flag set to true)


struct conditional_MAF{T <: NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
  layers::T
  conditional_num::Int
end

function conditional_MAF(layers...; conditional_num = 0)
  names = ntuple(i -> Symbol("MADE_$i"), length(layers))
  return conditional_MAF(NamedTuple{names}(layers), conditional_num)
end

(c::conditional_MAF)(x, ps, st::NamedTuple) = applyconditional_MAF(c.layers, x[1:end-c.conditional_num,:], ps, st, x[end-c.conditional_num+1:end,:])

@generated function applyconditional_MAF(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}, conditionals) where {fields}
N = length(fields) #number of MADE layers

x_symbols = vcat([:x], [gensym() for _ in 1:N])
total_std = [gensym() for _ in 1:N]
st_symbols = [gensym() for _ in 1:N]
calls1 = [:(($(x_symbols[i + 1]), $(st_symbols[i]), $(total_std[i])) = expr_forward(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]), conditionals)) for i in 1:N]
#calls2 = [:($(x_symbols[i]) = coord_transform($(x_symbols[i-1]),$(x_symbols[i]))) for i in 2:N]
#=push!(calls1, :(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[N]),
$(x_symbols[N]), ps.$(fields[N]), st.$(fields[N]))))=#

#add up all the log std for each layer
#each_layer_ouptut = :([$(x_symbols[N]) for i in 1:N])

################# add the definition of total_std as an array in the list of blocks
#calls[1] .= :($(x_symbols[1]) = 1)

push!(calls1, :(($(x_symbols[N + 1]), $(st_symbols[N])) = Lux.apply(layers.$(fields[N]),
$(x_symbols[N]), ps.$(fields[N]), st.$(fields[N]))))
push!(calls1, :($(total_std[N]) = copy($(x_symbols[N+1]))))
push!(calls1, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls1, :(return $(x_symbols[N + 1]), st, $(x_symbols[N]), $(total_std...)))
return Expr(:block, calls1...)
end

function expr_forward(layer::MADE, input, ps, st, conditionals)
  output, output_st  = Lux.apply(layer, input, ps,st)
  output_pre = copy(output)
  output = coord_transform(input, output)
  return(output, output_st, output_pre)
end

function expr_forward(layer::BatchNorm, input, ps, st, conditionals)
  output, output_st  = Lux.apply(layer, input, ps,st)
  output_pre = zeros(eltype(output), size(output))
  return(output, output_st, output_pre)
end


function expr_forward(layer::conditional_MADE, input, ps, st, conditionals)
  input = vcat(input, conditionals)
  output, output_st  = Lux.apply(layer, input, ps,st)
  output = coord_transform(input, output)
  return(output, output_st)
end


function sample(T::conditional_MAF, ps, st)
  _sample = randn(T.layers[1].layers[1].in_dims)
  for i in reverse(eachindex(T.layers))
    _sample = sample(T.layers[i], ps[i], st[i], samples = _sample)
  end
  return _sample
end