using Test

@testset "CombinatorialMaps" begin
  include("CombinatorialMaps.jl")
end

@testset "SimplicialSets" begin
  include("SimplicialSets.jl")
  include("DualSimplicialSets.jl")
end

@testset "Meshes" begin
  include("MeshInterop.jl")
  include("MeshGraphics.jl")
end
