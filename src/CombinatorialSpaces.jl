module CombinatorialSpaces

using Reexport

include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("ExteriorCalculus.jl")
include("SimplicialSets.jl")
include("DiscreteExteriorCalculus.jl")
include("MeshInterop.jl")
include("FastDEC.jl")
include("FastDualMesh.jl")
include("Meshes.jl")

@reexport using .SimplicialSets
@reexport using .DiscreteExteriorCalculus
@reexport using .FastDEC
@reexport using .FastDualMesh
@reexport using .MeshInterop
@reexport using .Meshes

end
