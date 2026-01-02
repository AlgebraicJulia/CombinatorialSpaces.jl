
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

dual_derivative!(res_dd0, Val(0), f; padding = 1)
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


# # TODO: Implement hodge 2
# f = tensorfy(Val(2), s, ones(nquads(s)))
# res_hdg2 = init_tensor_d(Val(0), s)

# hodge_star!(res_hdg2, Val(2), s, f)
# @test all(detensorfy_d(Val(0), s, res_hdg2) .== 0.5)

# hodge_star!(res_hdg2, Val(2), s, f; inv = true)
# @test all(detensorfy_d(Val(0), s, res_hdg2) .== 2)

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

all(detensorfy_d(Val(1), s, res_wdgdd01) .== 1)

# # Sharp PD
# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = 10

# X, Y = sharp_pd(s, alpha)

# @test X[1] == 1.0 && Y[1] == 0.0

# alpha = zeros(ne(s))
# alpha[3] = alpha[4] = -20

# X, Y = sharp_pd(s, alpha)

# @test X[1] == 0.0 && Y[1] == -2.0

# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = 10
# alpha[3] = alpha[4] = -20

# X, Y = sharp_pd(s, alpha)

# @test X[1] == 1.0 && Y[1] == -2.0

# # Sharp DD
# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = -10

# X, Y = sharp_dd(s, alpha)
# @test X[1] == 0.0 && Y[1] == 2.0

# alpha = zeros(ne(s))
# alpha[3] = alpha[4] = 10

# X, Y = sharp_dd(s, alpha)
# @test X[1] == 2.0 && Y[1] == 0.0

# alpha = zeros(ne(s))
# alpha[1] = alpha[2] = -10
# alpha[3] = alpha[4] = 10

# X, Y = sharp_dd(s, alpha)
# @test X[1] == 2.0 && Y[1] == 2.0

# # Exterior derivative
# d0 = exterior_derivative(Val(0), s)
# d1 = exterior_derivative(Val(1), s)

# @test all(0 .== d1 * d0)

# dual_d0 = dual_derivative(Val(0), s)
# dual_d1 = dual_derivative(Val(1), s)

# @test all(0 .== dual_d1 * dual_d0)

# # Wedge product
# v1 = ones(nv(s))
# e1 = ones(ne(s))
# q1 = ones(nquads(s))

# all(1.0 .== wedge_product(Val((0,1)), s, v1, e1))
# all(1.0 .== wedge_product(Val((0,2)), s, v1, q1))

# all(0.0 .== wedge_product(Val((1,1)), s, e1, e1))
# all(0.0 .== wedge_product(Val((1,1)), s, e1, 10 * e1))

# eX = zeros(ne(s))
# eX[1:nhe(s)] .= 1

# eY = zeros(ne(s))
# eY[nhe(s)+1:ne(s)] .= 1
# all(1.0 .== wedge_product(Val((1,1)), s, eX, eY))
# all(-1.0 .== wedge_product(Val((1,1)), s, eY, eX))

# # Hodge star
# hdg_0 = hodge_star(Val(0), s)
# @test (100 == sum(diag(hdg_0)))

# hdg_1 = hodge_star(Val(1), s)
# for i in 1:4 # Bottom horizontal edges
#   @test 1/(2*i) == diag(hdg_1)[i]
# end

# for i in 2:4 # Bottom interior vertical edges
#   @test 0.5*(2i-1) == diag(hdg_1)[i+nhe(s)]
# end

# hdg_2 = hodge_star(Val(2), s)
# @test 1 == diag(hdg_2)[1]
# @test 1/4 == diag(hdg_2)[6]
# @test 1/9 == diag(hdg_2)[11]
# @test 1/16 == diag(hdg_2)[16]