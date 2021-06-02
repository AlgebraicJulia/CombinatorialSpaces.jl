module CombinatorialSpaces

using Reexport
using Requires

include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("SimplicialSets.jl")
include("DiscreteExteriorCalculus.jl")
include("MeshInterop.jl")

function __init__()
  @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("MeshGraphics.jl")
end

@reexport using .SimplicialSets
@reexport using .DiscreteExteriorCalculus
@reexport using .MeshInterop

end
