using Test

@testset "CombinatorialMaps" begin
  include("CombinatorialMaps.jl")
end

@testset "SimplicialSets" begin
  include("SimplicialSets.jl")
  include("DualSimplicialSets.jl")
end

@testset "Utility Functions" begin
  include("MeshUtils.jl")
end

@testset "Visualization" begin
  include("Visualization.jl")
end
