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

@reexport using .SimplicialSets
@reexport using .DiscreteExteriorCalculus
@reexport using .FastDEC
@reexport using .MeshInterop

function __init__()
  @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("MeshGraphics.jl")
end

end
