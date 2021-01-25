""" Utitilies for working with dense and sparse arrays uniformly.
"""
module ArrayUtils
export enumeratenz, applynz, fromnz

using SparseArrays

abstract type AbstractArrayBuilder{T,N} end

struct ArrayBuilder{T,N,Arr<:AbstractArray{T,N}} <: AbstractArrayBuilder{T,N}
  array::Arr
end
add!(builder::ArrayBuilder, i, v) = builder.array[i] += v
take(builder::ArrayBuilder) = builder.array

struct SparseVectorBuilder{Tv,Ti<:Integer} <: AbstractArrayBuilder{Tv,1}
  n::Int
  I::Vector{Ti}
  V::Vector{Tv}
end
function add!(builder::SparseVectorBuilder, i::Integer, v)
  push!(builder.I, i); push!(builder.V, v)
end
function add!(builder::SparseVectorBuilder, i::AbstractVector, v)
  append!(builder.I, i); append!(builder.V, v)
end
take(builder::SparseVectorBuilder) = sparsevec(builder.I, builder.V, builder.n)

""" Object for efficient incremental construction of dense or sparse vector.
"""
nzbuilder(::Type{Arr}, dims::Integer...) where Arr =
  ArrayBuilder(zeros(Arr, dims))
nzbuilder(::Type{SparseVector{Tv,Ti}}, n::Integer) where {Tv,Ti} =
  SparseVectorBuilder(n, Ti[], Tv[])
nzbuilder(::Type{SparseVector{Tv}}, n::Integer) where Tv =
  SparseVectorBuilder(n, Int[], Tv[])

""" Enumerate structural nonzeros of vector.
"""
enumeratenz(v::AbstractVector) = enumerate(v)
enumeratenz(v::SparseVector) = zip(findnz(v)...)

""" Apply linear map defined in terms of structural nonzero values.
"""
function applynz(f, u::Vec, n::Integer) where Vec <: AbstractVector
  vec = nzbuilder(Vec, n)
  for (i, c) in enumeratenz(u)
    if !iszero(c)
      j, x = f(i)
      add!(vec, j, c*x)
    end
  end
  take(vec)
end

""" Construct array, not necessarily sparse, from structural nonzero values.
"""
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

end
