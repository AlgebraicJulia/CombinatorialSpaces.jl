module TestVisualization
using Test

using CairoMakie
using CombinatorialSpaces
using CombinatorialSpaces.Visualization
using StaticArrays
const Point3D = SVector{3,Float64}

CairoMakie.AbstractPlotting.inline!(true)

s = EmbeddedDeltaSet2D("assets/square.obj")
sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s)
subdivide_duals!(sd, Barycenter())

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
