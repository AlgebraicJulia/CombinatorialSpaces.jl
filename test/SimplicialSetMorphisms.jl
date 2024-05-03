module TestSimplicialSetMorphisms
using Test

using SparseArrays, StaticArrays

using Catlab.CategoricalAlgebra.CSets
using ACSets
using CombinatorialSpaces
using CombinatorialSpaces.Meshes: grid_345
using GeometryBasics: Point2, Point3
using LinearAlgebra: I

const Point2D = SVector{2,Float64}
const Point3D = SVector{3,Float64}

@testset "Simplicial Set Morphisms" begin
# Path graph on 3 vertices with irregular lengths.
s = EmbeddedDeltaSet1D{Bool,Point2D}()
add_vertices!(s, 3, point=[Point2D(1,0), Point2D(0,0), Point2D(0,2)])
add_edges!(s, [1,2], [2,3], edge_orientation=true)
t = EmbeddedDeltaSet1D{Bool,Point3D}()
add_vertices!(t, 3, point=[Point3D(1,0,0), Point3D(0,0,1), Point3D(0,2,2)])
add_edges!(t, [1,2], [2,3], edge_orientation=true)
sd = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(s)
subdivide_duals!(sd, Barycenter())

# Grid of 3,4,5 triangles.
t, td = grid_345()

# Grid of 3,4,5 triangles.
u, ud = grid_345()

Δst = ΔMap(
  Simplex{0}(1),
  Simplex{2}(3),
  SMatrix{1,3,Float64}([1/3 1/3 1/3]))

Δtu = ΔMap(
  Simplex{2}(3),
  Simplex{2}(1),
  SMatrix{3,3,Float64}(I(3)))

Δsu = compose(Δst, Δtu)

@test Δsu ==  ΔMap(
  Simplex{0}(1),
  Simplex{2}(1),
  SMatrix{1,3,Float64}([1/3 1/3 1/3] * I(3)))

end

ω = SimplexForm{0}([3, 4, 5]) #form on Δ²
ψ = pullback(Δtu,ω)
φ = pullback(Δst,ψ)

@test ω.data ≈ ψ.data

@test only(φ.data) ≈ 4

end #TestSimplicialSetMorphisms
