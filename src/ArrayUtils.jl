""" Array utilities.

- Dense and sparse arrays with a uniform interface
- Lazy operations on arrays
- Wapper structs for arrays to enable "typed" array computing
"""
module ArrayUtils
export @parts_array_struct, @vector_struct, applydiag, applynz, fromnz,
  enumeratenz, lazy

using LazyArrays: ApplyArray
using SparseArrays

# Dense and sparse arrays
#########################

abstract type AbstractArrayBuilder{T,N} end

struct ArrayBuilder{T,N,Arr<:AbstractArray{T,N}} <: AbstractArrayBuilder{T,N}
  array::Arr
end
add!(builder::ArrayBuilder, v, inds...) = builder.array[inds...] += v
take(builder::ArrayBuilder) = builder.array

struct SparseVectorBuilder{Tv,Ti<:Integer} <: AbstractArrayBuilder{Tv,1}
  n::Int
  I::Vector{Ti}
  V::Vector{Tv}
end
function add!(builder::SparseVectorBuilder, v, i::Integer)
  push!(builder.I, i); push!(builder.V, v)
end
function add!(builder::SparseVectorBuilder, v, i::AbstractVector)
  append!(builder.I, i); append!(builder.V, v)
end
take(builder::SparseVectorBuilder) = sparsevec(builder.I, builder.V, builder.n)

struct SparseMatrixBuilder{Tv,Ti<:Integer} <: AbstractArrayBuilder{Tv,2}
  m::Int
  n::Int
  I::Vector{Ti}
  J::Vector{Ti}
  V::Vector{Tv}
end
function add!(builder::SparseMatrixBuilder, v, i::Integer, j::Integer)
  push!(builder.I, i); push!(builder.J, j); push!(builder.V, v)
end
function add!(builder::SparseMatrixBuilder, v, i::AbstractVector, j::AbstractVector)
  append!(builder.I, i); append!(builder.J, j); append!(builder.V, v)
end
take(builder::SparseMatrixBuilder) = sparse(builder.I, builder.J, builder.V,
                                            builder.m, builder.n)

""" Object for efficient incremental construction of dense or sparse vector.
"""
nzbuilder(::Type{Arr}, dims::Integer...) where Arr =
  ArrayBuilder(zeros(Arr, dims))
nzbuilder(::Type{<:SparseVector{Tv}}, n::Integer) where Tv =
  SparseVectorBuilder(n, Int[], Tv[])
nzbuilder(::Type{<:SparseMatrixCSC{Tv}}, m::Integer, n::Integer) where {Tv,Ti} =
  SparseMatrixBuilder(m, n, Int[], Int[], Tv[])

""" Apply diagonal operator to dense or sparse vector.
"""
applydiag(f, x::AbstractVector) = map(f, eachindex(x), x)

function applydiag(f, x::SparseVector)
  I, V = findnz(x)
  sparsevec(I, map(f, I, V), length(x))
end

""" Apply linear map defined in terms of structural nonzero values.
"""
function applynz(f, x::Vec, m::Integer, n::Integer) where Vec <: AbstractVector
  length(x) == n ||
    error("Vector length does not match domain: $(length(x)) != $n")
  y = nzbuilder(Vec, m)
  for (i, a) in enumeratenz(x)
    if !iszero(a)
      j, b = f(i)
      add!(y, a*b, j)
    end
  end
  take(y)
end

""" Construct array, not necessarily sparse, from structural nonzero values.
"""
function fromnz(f, ::Type{Mat}, m::Integer, n::Integer) where Mat <: AbstractMatrix
  A = nzbuilder(Mat, m, n)
  for j in 1:n
    for (i, a) in zip(f(j)...)
      add!(A, a, i, j)
    end
  end
  take(A)
end

function fromnz(::Type{Vec}, I::AbstractVector, V::AbstractVector,
                n::Integer) where Vec <: AbstractVector
  vec = zeros(Vec, n)
  vec[I] = V
  vec
end
fromnz(::Type{<:SparseVector}, I::AbstractVector,
       V::AbstractVector, n::Integer) = sparsevec(I, V, n)

""" Enumerate structural nonzeros of vector.
"""
enumeratenz(v::AbstractVector) = enumerate(v)
enumeratenz(v::SparseVector) = zip(findnz(v)...)

""" Alternative to `Base.zeros` supporting dense and sparse arrays.
"""
zeros(::Type{Arr}, dims::Integer...) where Arr = zeros(Arr, dims)
zeros(::Type{Array{T,N}}, dims::Tuple{Vararg{Integer,N}}) where {T,N} =
  Base.zeros(T, dims)
zeros(::Type{<:SparseVector{T}}, dims::Tuple{Vararg{Integer,1}}) where T =
  spzeros(T, dims...)
zeros(::Type{<:SparseMatrixCSC{T}}, dims::Tuple{Vararg{Integer,2}}) where T =
  spzeros(T, dims...)

# Lazy arrays
#############

""" Perform operation lazily on arrays.
"""
lazy(::typeof(vcat), args...) = ApplyArray(vcat, args...)

# Wrapped arrays
################

""" Generate struct for a named vector struct.
"""
macro vector_struct(struct_sig)
  name, params = parse_struct_signature(struct_sig)
  :T ∉ params || error("Type parameter `T` is reserved")
  expr = quote
    Core.@__doc__ struct $(name){$(params...),T,V<:AbstractVector{T}} <: AbstractVector{T}
      data::V
    end
    $(if !isempty(params); quote
        $(name){$(params...)}(v::V) where {$(params...),T,V<:AbstractVector{T}} =
          $(name){$(params...),T,V}(v)
      end end)
    Base.size(v::$name) = size(v.data)
    Base.getindex(v::$name, i::Int) = getindex(v.data, i)
    Base.setindex!(v::$name, x, i::Int) = setindex!(v.data, x, i)
    Base.IndexStyle(::Type{<:$name}) = IndexLinear()
  end
  esc(expr)
end

""" Generate struct for C-set part or parts of a certain type.

The struct wraps an array and is mainly useful for dispatching on part type.
Example usage:

```julia
@parts_struct V
@parts_struct E

E(1)     # Edge #1 (zero-dimensional array)
V([1,4]) # Vertices #2 and #4 (one-dimensional array)
```
"""
macro parts_array_struct(struct_sig)
  name, params = parse_struct_signature(struct_sig)
  :N ∉ params || error("Type parameter `N` is reserved")
  expr = quote
    Core.@__doc__ struct $(name){$(params...),N,Data} <: AbstractArray{Int,N}
      data::Data
      $(name){$(params...)}(x::Int) where {$(params...)} =
        new{$(params...),0,Int}(x)
      $(name){$(params...)}(xs::Arr) where {$(params...),N,Arr<:AbstractArray{Int,N}} =
        new{$(params...),N,Arr}(xs)
    end
    Base.size(A::$name) = size(A.data)
    Base.getindex(A::$name, args...) = getindex(A.data, args...)
  end
  esc(expr)
end

function parse_struct_signature(expr)
  if expr isa Expr
    expr.head == :curly || error("Invalid struct expression: $expr")
    (expr.args[1]::Symbol, collect(Symbol, expr.args[2:end]))
  else
    (expr::Symbol, Symbol[])
  end
end

end
