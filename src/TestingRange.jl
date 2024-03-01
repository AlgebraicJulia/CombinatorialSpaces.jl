using CombinatorialSpaces
using CombinatorialSpaces: CombinatorialSpaces.DiscreteExteriorCalculus.FastMesh, CombinatorialSpaces.SimplicialSets.CayleyMengerDet
import CombinatorialSpaces: CombinatorialSpaces.DiscreteExteriorCalculus.make_dual_simplices_1d!, CombinatorialSpaces.DiscreteExteriorCalculus.make_dual_simplices_2d!
import CombinatorialSpaces: CombinatorialSpaces.SimplicialSets.negate, CombinatorialSpaces.DiscreteExteriorCalculus.relative_sign
using CombinatorialSpaces: CombinatorialSpaces.ArrayUtils.lazy
using ACSets.DenseACSets: attrtype_type

using GeometryBasics: Point3, Point2
using BenchmarkTools
using Catlab
using StaticArrays

s = triangulated_grid(100, 100, 1, 1, Point3{Float64})

begin
    sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}()
    copy_parts!(sd, s)
end

#= 
function test(s)
    sd = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3{Float64}}(s)
    subdivide_duals!(sd, Barycenter())
    sd
end

function fast_test(s)
    sd_c = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3{Float64}}(s, FastMesh())
    subdivide_duals!(sd_c, FastMesh(), Barycenter())
    sd_c
end

@info "Original Dual Mesh Generation"
test(s);
@time sd = test(s);

@info "New Dual Mesh Generation"
fast_test(s);
@time sd_c = fast_test(s);

sd == sd_c =#
    

# sd = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}(s)
# sd_c = EmbeddedDeltaDualComplex2D{Bool, Float64, Point3{Float64}}(s)

# subdivide_duals!(sd, Barycenter())
# subdivide_duals!(sd_c, FastMesh(), Barycenter())

# sd_c == sd
