module CombinatorialSpaces

using Reexport

include("Tries.jl")
include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("ExteriorCalculus.jl")
include("SimplicialSets.jl")
include("DiscreteExteriorCalculus.jl")
include("MeshInterop.jl")
include("FastDEC.jl")
include("Meshes.jl")
include("SimplicialComplexes.jl")
include("Multigrid.jl")



@reexport using .Tries
@reexport using .SimplicialSets
@reexport using .SimplicialComplexes
@reexport using .DiscreteExteriorCalculus
@reexport using .FastDEC
@reexport using .MeshInterop
@reexport using .Meshes
@reexport using .Multigrid

end
