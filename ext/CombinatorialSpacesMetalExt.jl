module CombinatorialSpacesMetalExt

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using Metal
using KernelAbstractions
using AppleAccelerate
using SparseArrays
import AppleAccelerate: AAFactorization
import CombinatorialSpaces: cache_wedge,
  dec_boundary, dec_differential, dec_dual_derivative,
  dec_hodge_star, dec_inv_hodge_star

# Wedge Product
#--------------

function cache_wedge(::Val{m}, ::Val{n}, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p}, ::Val{:Metal}, arr_cons=MtlArray, cast_float=nothing) where {float_type,_p,m,n}
  cache_wedge(m, n, sd, float_type, arr_cons, cast_float)
end
function cache_wedge(::Val{m}, ::Val{n}, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p}, ::Val{:Metal}, arr_cons=MtlArray, cast_float=nothing) where {float_type,_p,m,n}
  cache_wedge(m, n, sd, float_type, arr_cons, cast_float)
end

# Boundary and Co-boundary
#-------------------------

# Metal does not support sparse GPU matrices; return CPU sparse matrices.
# On Apple Silicon (unified memory), CPU and GPU share the same physical memory.

"""    dec_boundary(n::Int, sd::HasDeltaSet, ::Val{:Metal})

Compute a boundary matrix as a CPU sparse matrix.
Metal does not support sparse GPU matrices; Apple Silicon unified memory makes
this transparent for small simulations.
"""
dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Val{:Metal}) where float_type =
  SparseMatrixCSC{float_type}(dec_boundary(n, sd))
dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Val{:Metal}) where float_type =
  SparseMatrixCSC{float_type}(dec_boundary(n, sd))

"""    dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D, ::Val{:Metal})

Compute a dual derivative matrix as a CPU sparse matrix.
"""
dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Val{:Metal}) where float_type =
  SparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))
dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Val{:Metal}) where float_type =
  SparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))

"""    dec_differential(n::Int, sd::HasDeltaSet, ::Val{:Metal})

Compute an exterior derivative matrix as a CPU sparse matrix.
"""
dec_differential(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Val{:Metal}) where float_type =
  SparseMatrixCSC{float_type}(dec_differential(n, sd))
dec_differential(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Val{:Metal}) where float_type =
  SparseMatrixCSC{float_type}(dec_differential(n, sd))

# Hodge Star
#-----------

"""    dec_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Val{:Metal})

Compute a Hodge star as a Metal array (diagonal) or CPU sparse matrix (geometric Hodge 1-form).
Metal only supports Float32 meshes; for diagonal Hodge stars, returns a Metal matrix.
The geometric Hodge 1-form returns a CPU sparse matrix due to Metal's lack of sparse GPU support.
"""
dec_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Val{:Metal}) =
  dec_hodge_star(Val(n), sd, h, Val(:Metal))

dec_hodge_star(::Val{n}, sd::EmbeddedDeltaDualComplex1D{Bool, Float32, _p} where _p, h::DiscreteHodge, ::Val{:Metal}) where n =
  MtlArray(dec_hodge_star(Val(n), sd, h))

dec_hodge_star(::Val{n}, sd::EmbeddedDeltaDualComplex2D{Bool, Float32, _p} where _p, h::DiscreteHodge, ::Val{:Metal}) where n =
  MtlArray(dec_hodge_star(Val(n), sd, h))

dec_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D{Bool, Float32, _p} where _p, ::GeometricHodge, ::Val{:Metal}) =
  SparseMatrixCSC{Float32}(dec_hodge_star(Val(1), sd, GeometricHodge()))

# Inverse Hodge Star
#-------------------

"""    dec_inv_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Val{:Metal})

Compute an inverse Hodge star as a Metal array (diagonal) or a direct sparse solver function
(geometric Hodge 1-form). Metal only supports Float32 meshes.
"""
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Val{:Metal}) =
  dec_inv_hodge_star(Val(n), sd, h, Val(:Metal))

dec_inv_hodge_star(::Val{n}, sd::EmbeddedDeltaDualComplex1D{Bool, Float32, _p} where _p, h::DiscreteHodge, ::Val{:Metal}) where n =
  MtlArray(dec_inv_hodge_star(Val(n), sd, h))

dec_inv_hodge_star(::Val{n}, sd::EmbeddedDeltaDualComplex2D{Bool, Float32, _p} where _p, h::DiscreteHodge, ::Val{:Metal}) where n =
  MtlArray(dec_inv_hodge_star(Val(n), sd, h))

"""    dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Val{:Metal})

Return a function that solves the inverse geometric Hodge star for primal 1-forms using
AppleAccelerate's sparse QR direct solver via `AAFactorization`. The Hodge matrix is negated
before factorization to match the sign convention for the inverse (⋆⁻¹ = -⋆ for 1-forms in 2D).
"""
function dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D{Bool, Float32, _p} where _p, ::GeometricHodge, ::Val{:Metal})
  hdg = -1 * dec_hodge_star(1, sd, GeometricHodge(), Val(:Metal))
  hdg_fac = AAFactorization(hdg)
  x -> hdg_fac \ Array(x)
end

end
