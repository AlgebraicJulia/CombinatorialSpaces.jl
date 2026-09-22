using Test

@testset "2D Uniform Cubical Tests" begin
    include("UniformMeshTests.jl")
    include("UniformKernel.jl")
end

@testset "2D Uniform Cubical Graphics" begin
    include("UniformMeshGraphics.jl")
end

@testset "3D Uniform Cubical Tests" begin
    include("UniformMesh3DTests.jl")
    include("UniformKernel3D.jl")
end
