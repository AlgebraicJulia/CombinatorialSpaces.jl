# Subdivision of Meshes

To use the multigrid solvers or conduct a convergence analysis,
you need to construct a sequence of meshes of finer resolution.
The easiest way to get such a sequence is to take a coarse mesh
and recuresively subdivide each triangle until you reach a desired
resolution. This resolution could be determined by measuring the length
of the edges, or the area of the triangles.

This page will show you how to construct these sequences.
And demonstrate what the difrerent options look like.

```@example subdivision
using CombinatorialSpaces
using GeometryBasics
using GeometryBasics: Point3d
using CairoMakie, StaticArrays
using Colors

Blue =  colorant"#578AB7"
Orange =  colorant"#E0A578"
Green =  colorant"#ADCF94"
Beige =  colorant"#F9EECB"
Teal =  colorant"#B3DBD9"
Purple =  colorant"#D7C9F1"
```

## Binary Subdivision

This subdivision takes each edge and replaces it with two edges by subdividing at the midpoint.
The resulting mesh looks like the Tri-Force from the Legend of Zelda video games.

```@example subdivision
s = EmbeddedDeltaSet2D{Float64, Point3d}()
add_vertices!(s, 3, point=[Point3d(0,0,0), Point3d(1,0,0), Point3d(0.5,0.866,0)])
glue_sorted_triangle!(s, 1,2,3)
t = binary_subdivision(s)
u = binary_subdivision(t)
v = binary_subdivision(u)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="Initial Triangle", aspect=1.25)
wireframe!(ax, s, linewidth=6, color=Blue)
f
```

The first subdivision

```@example subdivision
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="1 Level of Binary Subdivision", aspect=1.25)
wireframe!(ax, t, linewidth=4, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
```

The second subdivision.

```@example subdivision
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="2 Levels of Binary Subdivision", aspect=1.25)
wireframe!(ax, u, linewidth=4, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
```

The third subdivision.

```@example subdivision
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="3 Levels of Binary Subdivision", aspect=1.25)
wireframe!(ax, v, linewidth=4, color=Teal)
wireframe!(ax, u, linewidth=6, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=6, color=Blue)
f
```

The fourth subdivision.

```@example subdivision
s = EmbeddedDeltaSet2D{Float64, Point3d}()
add_vertices!(s, 3, point=[Point3d(0,0,0), Point3d(1,0,0), Point3d(0.5,0.866,0)])
glue_sorted_triangle!(s, 1,2,3)
t = binary_subdivision(s)
u = binary_subdivision(t)
v = binary_subdivision(u)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="Binary Subdivision", aspect=1.25)
wireframe!(ax, v, linewidth=2, color=Teal)
wireframe!(ax, u, linewidth=4, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=8, color=Blue)
f
```

## Cubic Subdivision

The next subdivision scheme replaces each edge with three edges at the 1/3 and 2/3 points along the edge.
This mesh looks more like the nuclear radiation symbol.

```@example subdivision

s = EmbeddedDeltaSet2D{Float64, Point3d}()
add_vertices!(s, 3, point=[Point3d(0,0,0), Point3d(1,0,0), Point3d(0.5,0.866,0)])
glue_sorted_triangle!(s, 1,2,3)
t = cubic_subdivision(s)
u = cubic_subdivision(t)
v = cubic_subdivision(u)
f = Figure()
ax = CairoMakie.Axis(f[1,1]; title="Cubic Subdivision", aspect=1.25)
wireframe!(ax, v, linewidth=2, color=Teal)
wireframe!(ax, u, linewidth=4, color=Green)
wireframe!(ax, t, linewidth=6, color=Orange)
wireframe!(ax, s, linewidth=8, color=Blue)
f
```