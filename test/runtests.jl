using Test

@testset "CombinatorialMaps" begin
  include("CombinatorialMaps.jl")
end

@testset "SimplicialSets" begin
  include("SimplicialSets.jl")
  include("DualSimplicialSets.jl")
end

@testset "Utility Functions" begin
  include("MeshInterop.jl")
end

@testset "MeshGraphics" begin
  include("MeshGraphics.jl")
end
