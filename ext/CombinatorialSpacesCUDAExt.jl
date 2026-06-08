module CombinatorialSpacesCUDAExt

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using CUDA
using CUDA.cuSPARSE
using KernelAbstractions
using Krylov
import CombinatorialSpaces: cache_wedge,
  dec_boundary, dec_differential, dec_dual_derivative,
  dec_hodge_star, dec_inv_hodge_star

# Backend configuration
const _BACKEND = :CUDA
const _DEFAULT_ARR_CONS = CuArray

_backend_typed_sparse(::Type{T}, mat) where T = CuSparseMatrixCSC{T}(mat)
_backend_sparse(mat) = CuSparseMatrixCSC(mat)
_backend_dense(arr) = CuArray(arr)

# Shared operator definitions
include("GPUBackendShared.jl")

# CUDA-specific: untyped HasDeltaSet fallbacks for sparse operators

"""    dec_boundary(n::Int, sd::HasDeltaSet, ::Val{:CUDA})

Compute a boundary matrix as a sparse CUDA matrix.
"""
dec_boundary(n::Int, sd::HasDeltaSet, ::Val{:CUDA}) =
  CuSparseMatrixCSC(dec_boundary(n, sd))

"""    dec_dual_derivative(n::Int, sd::HasDeltaSet, ::Val{:CUDA})

Compute a dual derivative matrix as a sparse CUDA matrix.
"""
dec_dual_derivative(n::Int, sd::HasDeltaSet, ::Val{:CUDA}) =
  CuSparseMatrixCSC(dec_dual_derivative(n, sd))

"""    dec_differential(n::Int, sd::HasDeltaSet, ::Val{:CUDA})

Compute an exterior derivative matrix as a sparse CUDA matrix.
"""
dec_differential(n::Int, sd::HasDeltaSet, ::Val{:CUDA}) =
  CuSparseMatrixCSC(dec_differential(n, sd))

# CUDA-specific: inverse geometric Hodge star using GMRES solver

"""    dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Val{:CUDA})

Return a function that computes the inverse geometric Hodge star of a primal 1-form via a GMRES solver.
"""
function dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Val{:CUDA})
  hdg = -1 * dec_hodge_star(1, sd, GeometricHodge(), Val(:CUDA))
  x -> Krylov.gmres(hdg, x, atol = 1e-14)[1]
end

end

