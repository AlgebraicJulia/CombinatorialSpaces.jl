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
  include("Meshes.jl")
  include("MeshInterop.jl")
  include("MeshGraphics.jl")
end

@testset "Alternate Backends" begin
  include("Backends.jl")
end

end
