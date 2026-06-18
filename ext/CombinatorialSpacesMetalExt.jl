module CombinatorialSpacesMetalExt

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using Metal
using AppleAccelerate
using KernelAbstractions
using SparseArrays
import AppleAccelerate: AAFactorization
import CombinatorialSpaces: cache_wedge,
  dec_boundary, dec_differential, dec_dual_derivative,
  dec_hodge_star, dec_inv_hodge_star

# Backend configuration
const _BACKEND = :Metal
const _DEFAULT_ARR_CONS = MtlArray

_backend_typed_sparse(::Type{T}, mat) where T = SparseMatrixCSC{T}(mat)
_backend_sparse(mat::SparseMatrixCSC) = mat
_backend_dense(arr) = MtlArray(arr)

# Shared operator definitions
include("GPUBackendShared.jl")

# Metal-specific: inverse geometric Hodge star using AppleAccelerate direct solver

"""    dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Val{:Metal})

Return a function that solves the inverse geometric Hodge star for primal 1-forms using
AppleAccelerate's sparse QR direct solver via `AAFactorization`. The Hodge matrix is negated
before factorization to match the sign convention (⋆⁻¹ = -⋆ for 1-forms in 2D).
AAFactorization requires Int64 column indices; the internal sparse matrix is converted
from Int32 before factorization.
"""
function dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D{Bool, Float32, _p} where _p, ::GeometricHodge, ::Val{:Metal})
  hdg = SparseMatrixCSC{Float32, Int64}(-1 * dec_hodge_star(1, sd, GeometricHodge(), Val(:Metal)))
  hdg_fac = AAFactorization(hdg)
  x -> hdg_fac \ Array(x)
end

end
