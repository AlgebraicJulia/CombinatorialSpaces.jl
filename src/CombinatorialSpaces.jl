module CombinatorialSpaces

using Reexport
using Requires

include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("SimplicialSets.jl")
include("DualSimplicialSets.jl")
include("MeshUtils.jl")

function __init__()
  @require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" include("Visualization.jl")
end

@reexport using .SimplicialSets
@reexport using .DualSimplicialSets

end
