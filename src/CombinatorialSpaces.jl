module CombinatorialSpaces

using Reexport
using Requires

include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("ExteriorCalculus.jl")
include("SimplicialSets.jl")
include("DiscreteExteriorCalculus.jl")
include("MeshInterop.jl")
include("FastDEC.jl")
include("Meshes.jl")

@reexport using .SimplicialSets
@reexport using .DiscreteExteriorCalculus
@reexport using .FastDEC
@reexport using .MeshInterop
@reexport using .Meshes

function __init__()
  @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("MeshGraphics.jl")
  @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("../examples/GPU simulations/Wedge_GPU.jl")
end

end
