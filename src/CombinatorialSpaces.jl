module CombinatorialSpaces

using Reexport

include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("SimplicialSets.jl")
include("DualSimplicialSets.jl")

@reexport using .SimplicialSets
@reexport using .DualSimplicialSets

end
