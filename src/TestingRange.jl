using CombinatorialSpaces
using CombinatorialSpaces: CombinatorialSpaces.DiscreteExteriorCalculus.FastMesh, CombinatorialSpaces.SimplicialSets.CayleyMengerDet
using GeometryBasics: Point3, Point2
using BenchmarkTools
using Catlab
using StaticArrays

s = triangulated_grid(100, 100, 1, 1, Point3{Float64})
sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}(s)
sd_c = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}(s)

subdivide_duals!(sd, Barycenter())
subdivide_duals!(sd_c, FastMesh(), Barycenter())

sd_c == sd
