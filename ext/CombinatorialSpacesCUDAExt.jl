module CombinatorialSpacesCUDAExt

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
using Krylov
import CombinatorialSpaces: cache_wedge,
  dec_boundary, dec_differential, dec_dual_derivative,
  dec_hodge_star, dec_inv_hodge_star

# Wedge Product
#--------------

# Cast all but the last argument to CuArray.
function cache_wedge(::Type{Tuple{m,n}}, sd, ::Type{Val{:CUDA}}) where {m,n}
  wedge_cache = cache_wedge(Tuple{m,n}, sd, Val{:CPU})
  (CuArray.(wedge_cache[1:end-1])..., wedge_cache[end])
end

# Boundary and Co-boundary
#-------------------------

"""    dec_boundary(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}})

Compute a boundary matrix as a sparse CUDA matrix.
"""
dec_boundary(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_boundary(n, sd))

dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_boundary(n, sd))
dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_boundary(n, sd))

"""    dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D, ::Type{Val{:CUDA}})

Compute a dual derivative matrix as a sparse CUDA matrix.
"""
dec_dual_derivative(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_dual_derivative(n, sd))

dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))
dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))


"""    dec_differential(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}})

Compute an exterior derivative matrix as a sparse CUDA matrix.
"""
dec_differential(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_differential(n, sd))

dec_differential(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_differential(n, sd))
dec_differential(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_differential(n, sd))

# Hodge Star
#-----------

"""    dec_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}})

Compute a Hodge star as a diagonal or generic sparse CUDA matrix.
"""
dec_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) = 
  dec_hodge_star(Val{n}, sd, h, Val{:CUDA})

dec_hodge_star(::Type{Val{n}}, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) where n =
  CuArray(dec_hodge_star(Val{n}, sd, h))

dec_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, h::GeometricHodge, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_hodge_star(Val{1}, sd, h))

"""    dec_inv_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}})

Compute an inverse Hodge star matrix as a diagonal CUDA matrix.
"""
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) =
  dec_inv_hodge_star(Val{n}, sd, h, Val{:CUDA})

dec_inv_hodge_star(::Type{Val{n}}, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) where n =
  CuArray(dec_inv_hodge_star(Val{n}, sd, h))

"""    function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Type{Val{:CUDA}})

Return a function that computes the inverse geometric Hodge star of a primal 1-form via a GMRES solver.
"""
function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Type{Val{:CUDA}})
  hdg = -1 * dec_hodge_star(1, sd, GeometricHodge(), Val{:CUDA})
  x -> Krylov.gmres(hdg, x, atol = 1e-14)[1]
end

end

