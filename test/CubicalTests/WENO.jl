using Test
using Distributions

include("../../src/CubicalComplexes.jl")

weno = UniformWENO{5, Float64}()
stencil = RectStencil{5, Float64}(1,1,1,1,1)
@test 1.0 == WENO(weno, stencil)
