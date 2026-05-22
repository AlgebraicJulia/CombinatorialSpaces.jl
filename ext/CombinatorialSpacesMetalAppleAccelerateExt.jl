module CombinatorialSpacesMetalAppleAccelerateExt

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using Metal
using AppleAccelerate
using Krylov
using SparseArrays
import AppleAccelerate: AAFactorization
import CombinatorialSpaces: dec_inv_hodge_star, dec_hodge_star

"""    dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Val{:Metal})

Return a function that solves the inverse geometric Hodge star for primal 1-forms using
AppleAccelerate's sparse QR direct solver via `AAFactorization`. When AppleAccelerate is
not loaded alongside Metal, fall back to Krylov GMRES (defined here so that the GeometricHodge
inverse is always available when this extension is the sole provider).
The Hodge matrix is negated before factorization to match the sign convention
(⋆⁻¹ = -⋆ for 1-forms in 2D).
"""
function dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D{Bool, Float32, _p} where _p, ::GeometricHodge, ::Val{:Metal})
  # AAFactorization requires Int64 column indices; convert from the Int32 used internally.
  hdg = SparseMatrixCSC{Float32, Int64}(-1 * dec_hodge_star(1, sd, GeometricHodge(), Val(:Metal)))
  hdg_fac = AAFactorization(hdg)
  x -> hdg_fac \ Array(x)
end

end
