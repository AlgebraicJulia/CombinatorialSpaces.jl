module ArrayUtils
export enumeratenz

using SparseArrays

""" Enumerate structural nonzeros of vector.
"""
enumeratenz(v::AbstractVector) = enumerate(v)
enumeratenz(v::SparseVector) = zip(findnz(v)...)

""" Alternative to `Base.zeros` supporting dense and sparse arrays.
"""
zeros(::Type{A}, dims::Integer...) where A = zeros(A, dims)
zeros(::Type{Array{T,N}}, dims::Tuple{Vararg{Integer,N}}) where {T,N} =
  Base.zeros(T, dims)
zeros(::Type{SparseVector{T}}, dims::Tuple{Vararg{Integer,1}}) where T =
  spzeros(T, dims...)
zeros(::Type{SparseMatrixCSC{T}}, dims::Tuple{Vararg{Integer,2}}) where T =
  spzeros(T, dims...)

end
