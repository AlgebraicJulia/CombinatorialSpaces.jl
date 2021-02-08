""" Array utilities.

- Uniform interface for dense and sparse arrays
- Lazy array operations
"""
module ArrayUtils
export enumeratenz, applynz, fromnz, lazy

using LazyArrays: ApplyArray
using SparseArrays

# Data types
############

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

# Functions
###########

""" Enumerate structural nonzeros of vector.
"""
enumeratenz(v::AbstractVector) = enumerate(v)
enumeratenz(v::SparseVector) = zip(findnz(v)...)

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

""" Alternative to `Base.zeros` supporting dense and sparse arrays.
"""
zeros(::Type{Arr}, dims::Integer...) where Arr = zeros(Arr, dims)
zeros(::Type{Array{T,N}}, dims::Tuple{Vararg{Integer,N}}) where {T,N} =
  Base.zeros(T, dims)
zeros(::Type{<:SparseVector{T}}, dims::Tuple{Vararg{Integer,1}}) where T =
  spzeros(T, dims...)
zeros(::Type{<:SparseMatrixCSC{T}}, dims::Tuple{Vararg{Integer,2}}) where T =
  spzeros(T, dims...)

""" Lazy array operations.
"""
lazy(::typeof(vcat), args...) = ApplyArray(vcat, args...)

end
