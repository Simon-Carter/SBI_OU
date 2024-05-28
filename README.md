# SBI_OU
Simple repository to explore the usage of SBI for baysian posterior estimation


## This is a secondary title to test vscodes markdown abilities

### Masked Dense Layer

We use Masked autoregressive flowsas the base for our simulation based inference. 


```

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

```