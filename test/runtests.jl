module RunTests
using Test
using TetGen

@testset "Simplicial Sets" begin
  include("SimplicialSets.jl")
end

@testset "Exterior Calculus" begin
  include("ExteriorCalculus.jl")
  include("DiscreteExteriorCalculus.jl")
  include("Operators.jl")
end

@testset "Unitful Operators" begin
  include("Unitful.jl")
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

@testset "Mesh Optimization" begin
  include("MeshOptimization.jl")
end

@testset "Multigrid" begin
  include("Multigrid.jl")
end

@testset "Combinatorial Maps" begin
  include("CombinatorialMaps.jl")
end

end
