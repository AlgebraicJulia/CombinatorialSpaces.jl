module TestUniformMeshGraphics
using Test

using Makie
using CombinatorialSpaces

Makie.inline!(true)

s = UniformCubicalComplex2D(6, 6, 10.0, 10.0)

# Test Graphs
###############

fig, ax, ob = wireframe(s)
@test fig isa Figure
wireframe!(s)
@test fig isa Figure
fig, ax, ob = mesh(s)
@test fig isa Figure
mesh!(s)
@test fig isa Figure
fig, ax, ob = scatter(s)
@test fig isa Figure
scatter!(s)
@test fig isa Figure

end
