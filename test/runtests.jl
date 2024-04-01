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

try
  using CUDA
catch exception
  @info "CUDA loading failed."
end

if(!isnothing(Base.get_extension(RunTests, :CombinatorialSpacesCUDAExt)))
  @testset "CUDA" begin
    include("OperatorsCUDA.jl")
  end
else
  @info "CUDA tests were not run."
end

end