module CombinatorialSpaces

using Reexport

include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("ExteriorCalculus.jl")
include("SimplicialSets.jl")
include("DiscreteExteriorCalculus.jl")
include("MeshInterop.jl")
include("Meshes.jl")
include("Restrictions.jl")
include("Multigrid.jl")
include("FastDEC.jl")

@reexport using .SimplicialSets
@reexport using .DiscreteExteriorCalculus
@reexport using .MeshInterop
@reexport using .Meshes
@reexport using .Multigrid
@reexport using .FastDEC

end
