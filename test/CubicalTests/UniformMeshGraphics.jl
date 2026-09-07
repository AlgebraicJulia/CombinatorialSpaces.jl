module TestUniformMeshGraphics
using Test

using Makie
using CombinatorialSpaces

Makie.inline!(true)

s = UniformCubicalComplex2D(6, 6, 10.0, 10.0)

fig, ax, ob = wireframe(s)
@test fig isa Figure
p = wireframe!(s)
@test p isa Makie.Wireframe
fig, ax, ob = mesh(s)
@test fig isa Figure
p = mesh!(s)
@test p isa Makie.Mesh
fig, ax, ob = scatter(s)
@test fig isa Figure
p = scatter!(s)
@test p isa Makie.Scatter

end
