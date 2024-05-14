module TestMeshGraphics
using Test

using CairoMakie
using CombinatorialSpaces
using StaticArrays: SVector

const Point3D = SVector{3,Float64}

CairoMakie.Makie.inline!(true)

s = EmbeddedDeltaSet2D(joinpath(@__DIR__, "assets", "square.obj"))
sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3D}(s)
subdivide_duals!(sd, Barycenter())

# Test Graphs
###############

for ds in [s, sd]
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
