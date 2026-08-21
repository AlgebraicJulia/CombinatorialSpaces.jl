using Test

@testset "2D Uniform Cubical Tests" begin
    include("UniformMesh.jl")
    include("UniformKernel.jl")
end

@testset "3D Uniform Cubical Tests" begin
    include("UniformMesh3D.jl")
    include("UniformKernel3D.jl")
end