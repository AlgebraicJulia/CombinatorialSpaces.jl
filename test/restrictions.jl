using CombinatorialSpaces
using GeometryBasics: Point3
Point3D = Point3{Float64}

rect′ = loadmesh(Rectangle_30x10());
rect = EmbeddedDeltaDualComplex2D{Bool,Float64,Point3D}(rect′);
subdivide_duals!(rect, Barycenter());

left_wall_idxs(sd) = begin
    min_y = minimum(p -> p[2], sd[:point])
    findall(p -> abs(p[2] - min_y) ≤ sd[1,:length]+1e-4, sd[:point])
end

# 0-form
zero_form = 2*ones(nv(rect));

@test restrict(left_wall_idxs(rect), zero_form) == fill(2.0, 106)
@test restrict(rect, left_wall_idxs, zero_form) == fill(2.0, 106)

copy_zero_form = copy(zero_form);
mask!(rect, left_wall_idxs, copy_zero_form, fill(1114.0, 106));

@test copy_zero_form[copy_zero_form .== 1114.0] == fill(1114.0, 106)

