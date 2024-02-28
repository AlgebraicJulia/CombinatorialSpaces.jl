using CombinatorialSpaces
using CombinatorialSpaces: CombinatorialSpaces.DiscreteExteriorCalculus.FastMesh, CombinatorialSpaces.SimplicialSets.CayleyMengerDet
import CombinatorialSpaces: CombinatorialSpaces.DiscreteExteriorCalculus.make_dual_simplices_1d!, CombinatorialSpaces.DiscreteExteriorCalculus.make_dual_simplices_2d!
import CombinatorialSpaces: CombinatorialSpaces.SimplicialSets.negate, CombinatorialSpaces.DiscreteExteriorCalculus.relative_sign
using CombinatorialSpaces: CombinatorialSpaces.ArrayUtils.lazy
using ACSets.DenseACSets: attrtype_type

using GeometryBasics: Point3, Point2
using BenchmarkTools
using Catlab
using StaticArrays

s = triangulated_grid(100, 100, 1, 1, Point3{Float64})

sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}()
copy_parts!(sd, s)

# sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}(s)
# sd_c = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}(s)

# subdivide_duals!(sd, Barycenter())
# subdivide_duals!(sd_c, FastMesh(), Barycenter())

# sd_c == sd
