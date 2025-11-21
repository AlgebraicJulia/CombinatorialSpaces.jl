module CombinatorialSpaces

using Reexport

using GeometryBasics: Point2, Point3, Point2d, Point3d
const Point2D = Point2{Float64}
const Point3D = Point3{Float64}

export Point2, Point3, Point2d, Point3d, Point2D, Point3D

include("ArrayUtils.jl")
include("CombinatorialMaps.jl")
include("ExteriorCalculus.jl")
include("SimplicialSets.jl")
include("DiscreteExteriorCalculus.jl")
include("MeshInterop.jl")
include("CombMeshes.jl")
include("Restrictions.jl")
include("Multigrid.jl")
include("FastDEC.jl")
include("MeshOptimization.jl")

@reexport using .SimplicialSets
@reexport using .DiscreteExteriorCalculus
@reexport using .MeshInterop
@reexport using .CombMeshes
@reexport using .Multigrid
@reexport using .FastDEC
@reexport using .MeshOptimization

end
