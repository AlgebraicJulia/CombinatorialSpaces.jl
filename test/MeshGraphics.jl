module TestMeshGraphics
using Test

using Makie
using CombinatorialSpaces

Makie.inline!(true)

s = EmbeddedDeltaSet2D(joinpath(@__DIR__, "assets", "square.obj"))
sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3d}(s)
subdivide_duals!(sd, Barycenter())

# Test Graphs
###############

for ds in [s, sd]
  fig, ax, ob = wireframe(ds)
  @test fig isa Figure
  p = wireframe!(ds)
  @test p isa Makie.Wireframe
  fig, ax, ob = mesh(ds)
  @test fig isa Figure
  p = mesh!(ds)
  @test p isa Makie.Mesh
  fig, ax, ob = scatter(ds)
  @test fig isa Figure
  p = scatter!(ds)
  @test p isa Makie.Scatter
end

end
