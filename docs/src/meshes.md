# Meshes

```@autodocs
Modules = [ MeshInterop ]
Private = false
```

```@example 1
using JSServe # hide
Page(exportable=true, offline=true) # hide
```

## Graphical Rendering of DeltaSets

Below we have an example of how we render a mesh imported from a `.obj` file:
```@example 1
using FileIO, WGLMakie, CombinatorialSpaces
set_theme!(resolution=(800, 400))
catmesh = FileIO.load(File{format"OBJ"}(download("https://raw.githubusercontent.com/JuliaPlots/GLMakie.jl/master/src/GLVisualize/assets/cat.obj")))

catmesh_dset = EmbeddedDeltaSet2D(catmesh)
mesh(catmesh_dset, shading=false)
```

This can also be visualized as a wireframe:
```@example 1
wireframe(catmesh_dset)
```

We can also construct and plot the dual of this mesh
```@example 1
dual = EmbeddedDeltaDualComplex2D{Bool, Float32, Point{3, Float32}}(catmesh_dset)
subdivide_duals!(dual, Barycenter())
wireframe(dual)
```
