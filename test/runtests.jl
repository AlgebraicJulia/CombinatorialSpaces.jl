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

import Pkg
Pkg.add("CUDA")

try
  using CUDA
catch exception
  @info "CUDA tests were not run."
  @info CUDA.functional(true)
end

if CUDA.functional()
  @testset "CUDA" begin
    include("OperatorsCUDA.jl")
  end
end

end