module testGPU
#TODO: Should probably remove this test, moved to runtests itself
using Test
using CombinatorialSpaces
using Pkg

package_name = "CUDA"
is_cuda_installed = false

try
    using CUDA
    is_cuda_installed = true
catch e
    is_cuda_installed = false
end

if is_cuda_installed
    @testset "Conditional Loading of WedgeGPU.jl" begin
            @test isdefined(CombinatorialSpaces, dec_cu_c_wedge_product)
        end
    end
end