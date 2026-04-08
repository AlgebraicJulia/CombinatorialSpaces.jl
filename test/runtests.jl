module RunTests
using Test

@testset "CombinatorialMaps" begin
  include("CombinatorialMaps.jl")
end

@testset "SimplicialSets" begin
  include("SimplicialSets.jl")
end

@testset "ExteriorCalculus" begin
  include("ExteriorCalculus.jl")
  include("DiscreteExteriorCalculus.jl")
  include("Operators.jl")
end

@testset "Meshes" begin
  include("CombMeshes.jl")
  include("MeshInterop.jl")
  include("MeshGraphics.jl")
end

@testset "Alternate Backends" begin
  include("Backends.jl")
end

@testset "Restrictions" begin
  include("Restrictions.jl")
end

@testset "Multigrid" begin
  include("Multigrid.jl")
end

@testset "Mesh Optimization" begin
  include("MeshOptimization.jl")
end

end
