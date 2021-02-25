module TestVisualization
using Test

using CombinatorialSpaces
using CombinatorialSpaces.Visualization
using StaticArrays
using CairoMakie
const Point3D = SVector{3,Float64}

CairoMakie.AbstractPlotting.inline!(true)

# Import Tooling
################

s = EmbeddedDeltaSet2D("assets/square.obj")
@test s isa EmbeddedDeltaSet2D
@test triangle_vertices(s, 1) == [1,2,3]

sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s)
subdivide_duals!(sd, Barycenter())

# Test consistency with conversion to/from mesh
msh = CombinatorialSpaces.Visualization.make_mesh(s)
s_msh = EmbeddedDeltaSet2D(msh)
@test point(s_msh) == point(s)
@test triangle_vertices(s_msh) == triangle_vertices(s)

#@test triangle_vertices(s) == triangle_vertices(s_msh)
#@test point(s)

# Visualization
###############

for ds in [s ,sd]
  fig, ax, ob = wireframe(ds)
  @test fig isa Figure
  wireframe!(ds)
  @test fig isa Figure
  fig, ax, ob = mesh(ds)
  @test fig isa Figure
  mesh!(ds)
  @test fig isa Figure
  fig, ax, ob = scatter(ds)
  @test fig isa Figure
  scatter!(ds)
  @test fig isa Figure
end
end
