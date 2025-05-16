using CombinatorialSpaces
using CairoMakie

lx = 2
ly = 1
s = triangulated_grid(lx,ly,1,1,Point3d; shift = 0, diagonal = (x,y)->false)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
fig

s = triangulated_grid(lx,ly,1,1,Point3d; shift = 0, diagonal = (x,y)->true)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
fig

s = triangulated_grid(lx,ly,1,1,Point3d; shift = 0, diagonal = (x,y)->iseven(x))

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
fig

s = triangulated_grid(lx,ly,1,1,Point3d; shift = 0, diagonal = (x,y)->!iseven(x))

fig = Figure();
ax = CairoMakie.Axis(fig[1,1])
wireframe!(ax, s)
fig
