
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

function MaskedLinear(in_dims::Int, out_dims::Int; init_weight=glorot_uniform,
        init_bias=zeros32)
        init_mask=ones(Float32, out_dims, in_dims)
        init_mask_ref = Ref(init_mask)
  return MaskedLinear(in_dims, out_dims, init_weight, init_bias, init_mask_ref)
end

function Lux.initialparameters(rng::AbstractRNG, d::MaskedLinear)
    print("hi")
    return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
        bias=d.init_bias(rng, d.out_dims, 1))
end

function Lux.parameterlength(d::MaskedLinear)
    return d.out_dims * (d.in_dims + 1)
end

Lux.statelength(d::MaskedLinear) = 0

@inline function (d::MaskedLinear)(x::AbstractVecOrMat, ps, st::NamedTuple)
    return ((d.init_mask[]).*ps.weight)*x .+ ps.bias, st
end




# stuff

struct MADE{T <: NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
  layers::T
  mask::Base.RefValue{}
end

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

function generate_masks(m_k, gaussianMADE::Bool)
  Masks = []
  for i in eachindex(m_k[1:end-2]) 
    pair = collect(Iterators.product(m_k[i], m_k[i+1]))
    M = zeros(size(pair))
    foreach(i -> pair[i][1] > pair[i][2] ?  M[i] = 0 : M[i] = 1, CartesianIndices(pair))
    push!(Masks, M)
  end

  pair = collect(Iterators.product(m_k[end-1], m_k[end]))
  M = zeros(size(pair))
  foreach(i -> pair[i][1] >= pair[i][2] ?  M[i] = 0 : M[i] = 1, CartesianIndices(pair))

  if gaussianMADE
    M = hcat(M,M)
  end

  push!(Masks, M)

  return(Masks)
end

function MADE(layers...; gaussianMADE::Bool=true)
  names = ntuple(i -> Symbol("layer_$i"), length(layers))
  mask = generate_masks(generate_m_k(layers), gaussianMADE)
  mask_ref = Ref(mask)
  return MADE(NamedTuple{names}(layers), mask_ref)
end

function set_mask(layer::MaskedLinear, mask)
  layer.init_mask[] = mask'
end

(c::MADE)(x, ps, st::NamedTuple) = applyMADE(c.layers, x, ps, st, c.mask)

@generated function applyMADE(layers::NamedTuple{fields}, x, ps,
  st::NamedTuple{fields}, masks) where {fields}
N = length(fields)
x_symbols = vcat([:x], [gensym() for _ in 1:N])
st_symbols = [gensym() for _ in 1:N]

calls_mask = [:(set_mask(layers.$(fields[i]), $(:(masks[][$(i)])))) for i in 1:N]

calls = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
  $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))) for i in 1:N]

calls = vcat(calls_mask, calls)
push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st))
return Expr(:block, calls...)
end


MADE(; kwargs...) = MADE((; kwargs...))

#=
#quick test of the set_mask in applyMADE
@generated function test(layers::NamedTuple{fields}, masks) where {fields}
  N = length(fields)
  calls_mask = [:(set_mask(layers.$(fields[i]), $(:(masks[][$(i)])))) for i in 1:N]
  #calls_mask = [:($(println(i))) for i in 1:N]
  return Expr(:block, calls_mask...)
end
=#