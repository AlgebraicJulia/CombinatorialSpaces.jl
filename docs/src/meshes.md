# Meshes

```@example cat
using JSServe # hide
Page(exportable=true, offline=true) # hide
```

The two-dimensional embedded delta sets ([`EmbeddedDeltaSet2D`](@ref)) in
CombinatorialSpaces can be converted to and from mesh objects (`Mesh`) in
[Meshes.jl](https://github.com/JuliaGeometry/Meshes.jl). This is useful for
interoperation with packages in the
[JuliaGeometry](https://github.com/JuliaGeometry) ecosystem.

## Visualizing embedded delta sets

The following example shows how to import a mesh from an OBJ file, convert it
into an embedded delta set, and render it as a 3D mesh using WGLMakie.

```@example cat
using FileIO, WGLMakie, CombinatorialSpaces
set_theme!(resolution=(800, 400))
catmesh = FileIO.load(File{format"OBJ"}(download(
  "https://github.com/JuliaPlots/Makie.jl/raw/master/assets/cat.obj")))

catmesh_dset = EmbeddedDeltaSet2D(catmesh)
mesh(catmesh_dset, shading=false)
```

Alterntively, the embedded delta set can be visualized as a wireframe:

```@example cat
wireframe(catmesh_dset)
```

We can also construct and plot the dual complex for this mesh:

```@example cat
dual = EmbeddedDeltaDualComplex2D{Bool, Float32, Point{3,Float32}}(catmesh_dset)
subdivide_duals!(dual, Barycenter())
wireframe(dual)
```

## API docs

```@autodocs
Modules = [ MeshInterop ]
Private = false
```
