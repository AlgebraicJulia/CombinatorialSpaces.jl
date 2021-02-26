module TestMeshUtils
using Test

using CombinatorialSpaces
using CombinatorialSpaces.MeshUtils

using GeometryBasics
using StaticArrays
const Point3D = SVector{3,Float64}

# Import Tooling
################

s = EmbeddedDeltaSet2D("assets/square.obj")
@test s isa EmbeddedDeltaSet2D
@test triangle_vertices(s, 1) == [1,2,3]

sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s)
subdivide_duals!(sd, Barycenter())

# Test consistency with conversion to/from mesh
msh = make_mesh(s)
s_msh = EmbeddedDeltaSet2D(msh)
@test point(s_msh) == point(s)
@test triangle_vertices(s_msh) == triangle_vertices(s)

for i in [-2,-1,0,1,2]
  off_i = OffsetInteger{i, UInt32}(3)
  @test MeshUtils.convert(Int64, off_i) == 3
end
end
