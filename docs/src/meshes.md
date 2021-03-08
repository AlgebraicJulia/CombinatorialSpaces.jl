# Meshes

```@autodocs
Modules = [ Interop ]
Private = false
```

```@example
using FileIO, GLMakie, CombinatorialSpaces
catmesh = FileIO.load(GLMakie.assetpath("cat.obj"))
catmesh_dset = mesh_to_deltaset(catmesh)
gold = FileIO.load(download("https://raw.githubusercontent.com/nidorx/matcaps/master/1024/E6BF3C_5A4719_977726_FCFC82.png"))

mesh(catmesh_dset, matcap=gold, shading=false)
```
