module TestMeshInterop
using CombinatorialSpaces
using CombinatorialSpaces.Meshes: single_tetrahedron
using GeometryBasics
using GeometryBasics: Mesh, QuadFace, volume
using Test
using TetGen
using DelaunayTriangulation: triangulate, get_area, each_solid_triangle
using Random

# 2D
####

s_stl = EmbeddedDeltaSet2D(joinpath(@__DIR__, "assets", "square.stl"))
@test s_stl isa EmbeddedDeltaSet2D
@test triangle_vertices(s_stl, 1) == [1,2,3]

s = EmbeddedDeltaSet2D(joinpath(@__DIR__, "assets", "square.obj"))
@test s isa EmbeddedDeltaSet2D
@test triangle_vertices(s, 1) == [1,2,3]

sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3d}(s)
subdivide_duals!(sd, Barycenter())

# Test consistency with conversion to/from mesh.
msh = Mesh(s)
s_msh = EmbeddedDeltaSet2D(msh)
@test point(s_msh) == point(s)
@test triangle_vertices(s_msh) == triangle_vertices(s)

# 3D
####

# Construct a single tetrahedron natively.
s, sd = single_tetrahedron()
# Construct a single tetrahedron in GeometryBasics.
function single_tetrahedron_gb()
  pnts = connect([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], Point{3})
  Mesh(pnts, connect([1,2,3,4], SimplexFace{4}))
end
msh = single_tetrahedron_gb()

# Test equivalence of primal mesh & primal components:
@test Mesh(s) == msh
@test Mesh(sd, primal=true) == msh

# Test equivalence of dual mesh & dual components:
sd_to_mesh = Mesh(sd)
@test length(faces(sd_to_mesh)) == 24
@test sd_to_mesh.position == sd[:dual_point]

# TetGen compatibility
#---------------------
# Create mesh from the TetGen.jl/README.md.
# https://github.com/JuliaGeometry/TetGen.jl/blob/ea73adce3ea4dfa6062eb84b1eff05f3fcab60a5/README.md
function tetgen_readme_mesh()
  points = Point{3, Float64}[
    (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    (0.0, 0.0, 12.0), (2.0, 0.0, 12.0),
    (2.0, 2.0, 12.0), (0.0, 2.0, 12.0)]
  facets = QuadFace{Cint}[
    1:4, 5:8,
    [1,5,6,2],
    [2,6,7,3],
    [3, 7, 8, 4],
    [4, 8, 5, 1]]
  markers = Cint[-1, -2, 0, 0, 0, 0]
  mesh = GeometryBasics.MetaMesh(points, facets; markers)
  mesh, tetrahedralize(mesh, "Qvpq1.414a0.1");
end
msh, tet_msh = tetgen_readme_mesh()

# Import.
s = EmbeddedDeltaSet3D(tet_msh)
sd = EmbeddedDeltaDualComplex3D{Bool, Float64, Point3d}(s)
subdivide_duals!(sd, Circumcenter())
@test sum(sd[:vol]) ≈ 2*2*12
s[:edge_orientation] = true
s[:tri_orientation] = true
@test is_manifold_like(s)

# Test mesh roundtripping:
s_to_mesh = Mesh(s)
@test all(issetequal.(faces(s_to_mesh), faces(tet_msh)))

# DelaunayTriangulation compatibility
#---------------------

# Structured unit square
lx = ly = 1
dx = dy = 0.1
points = Point2d[]
for x in 0:dx:lx
  for y in 0:dy:ly
    push!(points, Point2d(x,y))
  end
end
tri = triangulate(points)
s = EmbeddedDeltaSet2D(tri)
@test all(Point2d.(s[:point]) .== points)
@test length(each_solid_triangle(tri)) == ntriangles(s)
@test first(s[:point]) isa Point3d

sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3d}(s)
subdivide_duals!(sd, Circumcenter())
@test sum(sd[:area]) ≈ get_area(tri)
@test is_manifold_like(s)

# Randomized unit square
Random.seed!(0)
corners = [Point2{Float32}(0,0), Point2{Float32}(0,1), Point2{Float32}(1,0), Point2{Float32}(1,1)]
points = vcat(rand(Point2{Float32}, 100), corners)

tri = triangulate(points)
s = EmbeddedDeltaSet2D(tri)

@test all(Point2d.(s[:point]) .== points)
@test length(each_solid_triangle(tri)) == ntriangles(s)
@test first(s[:point]) isa Point3{Float32}

sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3d}(s)
subdivide_duals!(sd, Circumcenter())
@test sum(sd[:area]) ≈ get_area(tri)
@test is_manifold_like(s)

end
