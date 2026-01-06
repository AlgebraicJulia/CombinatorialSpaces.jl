
include("../../src/CubicalComplexes.jl")

s = uniform_grid(10, 10, 2, 2)

# Exterior derivative
res_d0 = init_tensor(Val(1), s)
f = tensorfy(Val(0), s, [1, 2, 3, 4])
exterior_derivative!(res_d0, Val(0), f)

@test detensorfy(Val(1), s, res_d0) == Float64[1, 1, 2, 2]

res_d1 = init_tensor(Val(2), s)
f = tensorfy(Val(1), s, [2, -2, -2, 2])
exterior_derivative!(res_d1, Val(1), f)

@test detensorfy(Val(2), s, res_d1) == [8.0]

f = tensorfy(Val(0), s, [1, 2, 3, 4])
res_d0 = init_tensor(Val(1), s)
res_d1 = init_tensor(Val(2), s)
exterior_derivative!(res_d0, Val(0), f)
exterior_derivative!(res_d1, Val(1), res_d0)

@test detensorfy(Val(2), s, res_d1) == [0.0]

# Dual derivative
f = tensorfy_d(Val(0), s, [1])
res_dd0 = init_tensor_d(Val(1), s)
dual_derivative!(res_dd0, Val(0), s, f)

@test detensorfy(Val(1), s, res_dd0) == [-1, 1, 1, -1]

dual_derivative!(res_dd0, Val(0), s, f; padding = 1)
@test all(detensorfy(Val(1), s, res_dd0) .== 0)

f = tensorfy_d(Val(1), s, ones(nde(s)))
res_dd1 = init_tensor_d(Val(2), s)
dual_derivative!(res_dd1, Val(1), s, f)

@test detensorfy_d(Val(2), s, res_dd1) == [-2, 0, 0, 2]

# Hodge star
f = tensorfy(Val(0), s, ones(nv(s)))
res_hdg0 = init_tensor_d(Val(2), s)

hodge_star!(res_hdg0, Val(0), s, f)
@test all(detensorfy_d(Val(2), s, res_hdg0) .== 25)

hodge_star!(res_hdg0, Val(0), s, f; inv = true)
@test all(detensorfy_d(Val(2), s, res_hdg0) .== 1/25)


f = tensorfy(Val(1), s, ones(ne(s)))
res_hdg1 = init_tensor_d(Val(1), s)

hodge_star!(res_hdg1, Val(1), s, f)
@test all(detensorfy_d(Val(1), s, res_hdg1) .== 0.5)

hodge_star!(res_hdg1, Val(1), s, f; inv = true)
@test all(detensorfy_d(Val(1), s, res_hdg1) .== 2)

f = tensorfy(Val(2), s, ones(nquads(s)))
res_hdg2 = init_tensor_d(Val(0), s)

hodge_star!(res_hdg2, Val(2), s, f)
@test all(detensorfy_d(Val(0), s, res_hdg2) .== 100)

hodge_star!(res_hdg2, Val(2), s, f; inv = true)
@test all(detensorfy_d(Val(0), s, res_hdg2) .== 1/100)

# Wedge product
v1 = tensorfy(Val(0), s, [1, 2, 1, 2])
e1 = tensorfy(Val(1), s, ones(ne(s)))
q1 = tensorfy(Val(2), s, ones(nquads(s)))

res_wdg01 = init_tensor(Val(1), s)
wedge_product!(res_wdg01, Val((0,1)), s, v1, e1)
@test detensorfy(Val(1), s, res_wdg01) == [1.5, 1.5, 1.0, 2.0]

# # TODO: Implement wedge 0-2
# res_wdg02 = init_tensor(Val(2), s)
# wedge_product!(res_wdg02, Val((0,2)), s, v1, q1)
# @test detensorfy(Val(1), s, res_wdg01) == [1.5]

res_wdg11 = init_tensor(Val(2), s)
wedge_product!(res_wdg11, Val((1,1)), s, e1, e1)
@test detensorfy(Val(2), s, res_wdg11) == [0]

wedge_product!(res_wdg11, Val((1,1)), s, e1, 10 .* e1)
@test detensorfy(Val(2), s, res_wdg11) == [0]

eX = tensorfy(Val(1), s, [1, 1, 0, 0])
eY = tensorfy(Val(1), s, [0, 0, 1, 1])

wedge_product!(res_wdg11, Val((1,1)), s, eX, eY)
@test detensorfy(Val(2), s, res_wdg11) == [1]

wedge_product!(res_wdg11, Val((1,1)), s, eY, eX)
@test detensorfy(Val(2), s, res_wdg11) == [-1]

dv1 = tensorfy_d(Val(0), s, ones(ndv(s)))
de1 = tensorfy_d(Val(1), s, ones(nde(s)))

res_wdgdd01 = init_tensor_d(Val(1), s)

wedge_product_dd!(res_wdgdd01, Val((0,1)), s, dv1, de1)

@test all(detensorfy_d(Val(1), s, res_wdgdd01) .== 1)

# Sharp DD

f = tensorfy(Val(1), s, [-10, -10, 0, 0])

X = init_tensor_d(Val(0), s)
Y = init_tensor_d(Val(0), s)

sharp_dd!(X, Y, s, f)
@test X[1, 1] == 0.0 && Y[1, 1] == 2.0

f = tensorfy(Val(1), s, [0, 0, 10, 10])

sharp_dd!(X, Y, s, f)
@test X[1, 1] == 2.0 && Y[1, 1] == 0.0

f = tensorfy(Val(1), s, [-10, -10, 10, 10])

sharp_dd!(X, Y, s, f)
@test X[1, 1] == 2.0 && Y[1, 1] == 2.0

# Flat DP

res_flat_dp = init_tensor(Val(1), s)

X = [1]; Y = [0]
flat_dp!(res_flat_dp, s, X, Y)
@test all(xedges(res_flat_dp) .== 10)
@test all(yedges(res_flat_dp) .== 0)

X = [0]; Y = [1]
flat_dp!(res_flat_dp, s, X, Y)
@test all(xedges(res_flat_dp) .== 0)
@test all(yedges(res_flat_dp) .== 10)

X = [-1]; Y = [-1]
flat_dp!(res_flat_dp, s, X, Y)
@test all(xedges(res_flat_dp) .== -10)
@test all(yedges(res_flat_dp) .== -10)

### Test with irregular mesh w/ interior
ps = Point3d[]
for y in [0, 1, 3, 6, 10]
  for x in [0, 1, 3, 6, 10]
    push!(ps, Point3d(x, y, 0))
  end
end

s = EmbeddedCubicalComplex2D(5, 5, ps);

# Exterior derivative
res_d0 = init_tensor(Val(1), s)
f = tensorfy(Val(0), s, repeat([1, 2, 3, 4, 5], 5))
exterior_derivative!(res_d0, Val(0), f)

@test all(xedges(res_d0) .== 1)
@test all(yedges(res_d0) .== 0)

# res_d1 = init_tensor(Val(2), s)
# f = tensorfy(Val(1), s, [2, -2, -2, 2])
# exterior_derivative!(res_d1, Val(1), f)

# @test detensorfy(Val(2), s, res_d1) == [8.0]

# Dual derivative
f = repeat([1, 2, 3, 4], 1, 4)
res_dd0 = init_tensor_d(Val(1), s)
dual_derivative!(res_dd0, Val(0), s, f)

@test all(d_xedges(res_dd0)[2:4, :] .== 1)
@test all(d_xedges(res_dd0)[5, :] .== -4) # Due to boundary conditions

@test all(d_yedges(res_dd0)[:, 2:4] .== 0)
@test d_yedges(res_dd0)[:, 5] == [1, 2, 3, 4]
@test d_yedges(res_dd0)[:, 1] == [-1, -2, -3, -4]

dual_derivative!(res_dd0, Val(0), s, f; padding = 1)
@test all(d_yedges(res_dd0)[:, 2:4] .== 0)

@test all(d_xedges(res_dd0)[5, :] .== -3) # Due to boundary conditions

# Hodge star
f = tensorfy(Val(0), s, ones(nv(s)))
res_hdg0 = init_tensor_d(Val(2), s)

hodge_star!(res_hdg0, Val(0), s, f)
@test res_hdg0 == res_hdg0'

@test res_hdg0[5, 1] == 1.0
@test res_hdg0[5, 2] == 2.0 + 1.0
@test res_hdg0[5, 3] == 3.0 + 2.0

hodge_star!(res_hdg0, Val(0), s, f; inv = true)
@test res_hdg0[5, 5] == 1 / (4.0)

f = tensorfy(Val(1), s, ones(ne(s)))
res_hdg1 = init_tensor_d(Val(1), s)

hodge_star!(res_hdg1, Val(1), s, f)
@test d_yedges(res_hdg1)[4, 1] == 1/8
@test d_yedges(res_hdg1)[4, 2] == 1.5/4

@test d_xedges(res_hdg1)[1, 1] == 0.5
@test d_xedges(res_hdg1)[2, 1] == 1.5

f = tensorfy(Val(2), s, ones(nquads(s)))
res_hdg2 = init_tensor_d(Val(0), s)

qA = init_tensor(Val(2), s)
for q in CartesianIndices(qA)
  x, y = q.I
  qA[q] = quad_area(s, x, y)
end

hodge_star!(res_hdg2, Val(2), s, f)
@test all(res_hdg2 .== qA)

hodge_star!(res_hdg2, Val(2), s, f; inv = true)
@test all(res_hdg2 .== 1 ./ qA)

# Wedge product
v1 = tensorfy(Val(0), s, ones(nv(s)))
e1 = tensorfy(Val(1), s, ones(ne(s)))
q1 = tensorfy(Val(2), s, ones(nquads(s)))

res_wdg01 = init_tensor(Val(1), s)
wedge_product!(res_wdg01, Val((0,1)), s, v1, e1)
@test all(xedges(res_wdg01) .== 1)
@test all(yedges(res_wdg01) .== 1)

res_wdg11 = init_tensor(Val(2), s)
wedge_product!(res_wdg11, Val((1,1)), s, e1, e1)
@test all(res_wdg11 .== 0)

wedge_product!(res_wdg11, Val((1,1)), s, e1, 10 .* e1)
@test all(res_wdg11 .== 0)

eX = init_tensor(Val(1), s); xedges(eX) .= 1
eY = init_tensor(Val(1), s); yedges(eY) .= 1

wedge_product!(res_wdg11, Val((1,1)), s, eX, eY)
@test all(res_wdg11 .== 1)

wedge_product!(res_wdg11, Val((1,1)), s, eY, eX)
@test all(res_wdg11 .== -1)

eX = init_tensor(Val(1), s)
for e in CartesianIndices(xedges(eX))
  x, y = e.I
  xedges(eX)[e] = edge_len(s, 1, x, y)
end

eY = init_tensor(Val(1), s)
for e in CartesianIndices(yedges(eY))
  x, y = e.I
  yedges(eY)[e] = edge_len(s, 2, x, y)
end

wedge_product!(res_wdg11, Val((1,1)), s, eX, eY)

@test all(res_wdg11 .== qA)

dv1 = tensorfy_d(Val(0), s, ones(ndv(s)))
de1 = tensorfy_d(Val(1), s, ones(nde(s)))

res_wdgdd01 = init_tensor_d(Val(1), s)

wedge_product_dd!(res_wdgdd01, Val((0,1)), s, dv1, de1)

@test all(detensorfy_d(Val(1), s, res_wdgdd01) .== 1)

# Periodic conditions

s = uniform_grid(1, 1, 1001, 1001)

u_ten = tensorfy(s, map(p -> p[1], points(s)))

v_ten = init_tensor(Val(0), s)

boundary_v_map!(v_ten, s, u_ten, (1, 0)) # Only in x

@test v_ten[nx(s), :] == u_ten[2, :]
@test v_ten[1, :] == u_ten[nx(s) - 1, :]

u_ten = tensorfy(s, map(p -> p[2], points(s)))

v_ten = init_tensor(Val(0), s)

boundary_v_map!(v_ten, s, u_ten, (0, 1)) # Only in y

@test v_ten[:, nx(s)] == u_ten[:, 2]
@test v_ten[:, 1] == u_ten[:, nx(s) - 1]

# TODO: Investigate matrix-free solvers
# Matrix DEC (Needed for linear solves)

s = uniform_grid(10, 10, 2, 2)

d0_mat = exterior_derivative(Val(0), s)
d1_mat = exterior_derivative(Val(1), s)

@test all(d1_mat * d0_mat .== 0)
@test all(d0_mat * ones(nv(s)) .== 0)
@test all(d1_mat * ones(ne(s)) .== 0)

