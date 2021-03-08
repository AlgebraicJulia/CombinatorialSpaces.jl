# Meshes

```@autodocs
Modules = [ Interop ]
Private = false
```

```@example
using JSServe
Page(exportable=true, offline=true)

using FileIO, WGLMakie, CombinatorialSpaces
catmesh = FileIO.load(File{format"OBJ"}(download("https://raw.githubusercontent.com/JuliaPlots/GLMakie.jl/master/src/GLVisualize/assets/cat.obj")))
catmesh_dset = mesh_to_deltaset(catmesh)
gold = FileIO.load(download("https://raw.githubusercontent.com/nidorx/matcaps/master/1024/E6BF3C_5A4719_977726_FCFC82.png"))

mesh(catmesh_dset, matcap=gold, shading=false)
```
