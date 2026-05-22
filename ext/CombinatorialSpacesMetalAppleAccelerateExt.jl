module CombinatorialSpacesMetalAppleAccelerateExt

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using Metal
using AppleAccelerate
using SparseArrays
import AppleAccelerate: AAFactorization
import CombinatorialSpaces: dec_inv_hodge_star, dec_hodge_star

"""    dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Val{:Metal})

Return a function that solves the inverse geometric Hodge star for primal 1-forms using
AppleAccelerate's sparse QR direct solver via `AAFactorization`. This overrides the GMRES
fallback from `CombinatorialSpacesMetalExt` when AppleAccelerate is also loaded.
The Hodge matrix is negated before factorization to match the sign convention
(⋆⁻¹ = -⋆ for 1-forms in 2D).
"""
function dec_inv_hodge_star(::Val{1}, sd::EmbeddedDeltaDualComplex2D{Bool, Float32, _p} where _p, ::GeometricHodge, ::Val{:Metal})
  hdg = -1 * dec_hodge_star(1, sd, GeometricHodge(), Val(:Metal))
  hdg_fac = AAFactorization(hdg)
  x -> hdg_fac \ Array(x)
end

end
