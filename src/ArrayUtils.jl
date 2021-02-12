""" Array utilities.

- Dense and sparse arrays with a uniform interface
- Lazy operations on arrays
- Structs that wrap arrays for "typed" array computing
"""
module ArrayUtils
export @parts_array, enumeratenz, applynz, fromnz, lazy

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

# Lazy arrays
#############

""" Perform operation lazily on arrays.
"""
lazy(::typeof(vcat), args...) = ApplyArray(vcat, args...)

# Wrapped arrays
################

""" Abstract type for array of C-set parts of a certain type.
"""
abstract type AbstractPartsArray{N} <: AbstractArray{Int,N} end

""" Generate struct wrapping a C-set part or parts of a certain type.

Useful mainly for dispatching on part type. Example usage:

```julia
@parts_struct V
@parts_struct E

E(1)     # Edge with ID 1 (a zero-dimensional array)
V([1,4]) # Vertices with IDs 2 and 4 (a one-dimensional array)
```
"""
macro parts_array(type_expr)
  name, params = if type_expr isa Expr
    type_expr.head == :curly || error("Invalid type expression: $type_expr")
    (type_expr.args[1]::Symbol, collect(Symbol, type_expr.args[2:end]))
  else
    (type_expr::Symbol, Symbol[])
  end
  :N ∉ params || error("Type parameter `N` is reserved")
  supername = GlobalRef(ArrayUtils, :AbstractPartsArray)
  expr = quote
    Core.@__doc__ struct $(name){$(params...),N,Data} <: $(supername){N}
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

end
