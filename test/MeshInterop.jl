module TestMeshInterop
using CombinatorialSpaces
using GeometryBasics
using GeometryBasics: Mesh, QuadFace, volume
using StaticArrays: SVector
using Test
using TetGen

const Point3D = SVector{3,Float64}

# Import/ Export of meshes
##########################

# 2D
#---
s_stl = EmbeddedDeltaSet2D(joinpath(@__DIR__, "assets", "square.stl"))
@test s_stl isa EmbeddedDeltaSet2D
@test triangle_vertices(s_stl, 1) == [1,2,3]

s = EmbeddedDeltaSet2D(joinpath(@__DIR__, "assets", "square.obj"))
@test s isa EmbeddedDeltaSet2D
@test triangle_vertices(s, 1) == [1,2,3]

sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s)
subdivide_duals!(sd, Barycenter())

# Test consistency with conversion to/from mesh.
msh = Mesh(s)
s_msh = EmbeddedDeltaSet2D(msh)
@test point(s_msh) == point(s)
@test triangle_vertices(s_msh) == triangle_vertices(s)

# Test consistency with conversion to/from mesh.
msh = Mesh(s)
s_msh = EmbeddedDeltaSet2D(msh)
@test point(s_msh) == point(s)
@test triangle_vertices(s_msh) == triangle_vertices(s)

# TetGen Compatibility
######################

# TetGen.jl/README.md example:
# https://github.com/JuliaGeometry/TetGen.jl/blob/ea73adce3ea4dfa6062eb84b1eff05f3fcab60a5/README.md
#--------------------------------------
#"""
# Construct a cube out of Quads
points = Point{3, Float64}[
    (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    (0.0, 0.0, 12.0), (2.0, 0.0, 12.0),
    (2.0, 2.0, 12.0), (0.0, 2.0, 12.0)
]

facets = QuadFace{Cint}[
    1:4,
    5:8,
    [1,5,6,2],
    [2,6,7,3],
    [3, 7, 8, 4],
    [4, 8, 5, 1]
]

markers = Cint[-1, -2, 0, 0, 0, 0]
# attach some additional information to our faces!
mesh = Mesh(points, meta(facets, markers=markers))
result = tetrahedralize(mesh, "vpq1.414a0.1")
#"""
s = EmbeddedDeltaSet3D(result)
sd = EmbeddedDeltaDualComplex3D{Bool, Float64, Point3D}(s)
subdivide_duals!(sd, Circumcenter())
@test sum(sd[:vol]) ≈ 2*2*12

end
