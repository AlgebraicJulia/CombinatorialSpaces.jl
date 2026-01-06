using BenchmarkTools

include("../../src/CubicalComplexes.jl")

@benchmark uniform_grid(1, 1001)

s = uniform_grid(1, 1001)

### KERNELS ###

# Exterior derivatives

in_d0 = init_tensor(Val(0), s);
out_d0 = init_tensor(Val(1), s);

@benchmark exterior_derivative!(out_d0, Val(0), in_d0) # 457.555 μs

in_d1 = init_tensor(Val(1), s);
out_d1 = init_tensor(Val(2), s);

@benchmark exterior_derivative!(out_d1, Val(1), in_d1) # 865.002 μs


# Dual derivatives

in_dd0 = init_tensor_d(Val(0), s);
out_dd0 = init_tensor_d(Val(1), s);

@benchmark dual_derivative!(out_dd0, Val(0), s, in_dd0) # 5.530 ms

in_dd1 = init_tensor_d(Val(1), s);
out_dd1 = init_tensor_d(Val(2), s);

@benchmark dual_derivative!(out_dd1, Val(1), s, in_dd1) # 4.276 ms


# Hodge star

in_hdg0 = init_tensor(Val(0), s);
out_hdg0 = init_tensor(Val(0), s);

@benchmark hodge_star!(out_hdg0, Val(0), s, in_hdg0) # 479.208 ms
@benchmark hodge_star!(out_hdg0, Val(0), s, in_hdg0; inv = true) # 484.218 ms

in_hdg1 = init_tensor(Val(1), s);
out_hdg1 = init_tensor(Val(1), s);

@benchmark hodge_star!(out_hdg1, Val(1), s, in_hdg1) # 365.115 ms
@benchmark hodge_star!(out_hdg1, Val(1), s, in_hdg1; inv = true) # 366.264 ms

in_hdg2 = init_tensor(Val(2), s);
out_hdg2 = init_tensor(Val(2), s);

@benchmark hodge_star!(out_hdg2, Val(2), s, in_hdg2) # 121.338 ms
@benchmark hodge_star!(out_hdg2, Val(2), s, in_hdg2; inv = true) # 123.188 ms


# Wedge product

in_wdg01_0 = init_tensor(Val(0), s);
in_wdg01_1 = init_tensor(Val(1), s);
out_wdg01 = init_tensor(Val(1), s);

@benchmark wedge_product!(out_wdg01, Val((0,1)), s, in_wdg01_0, in_wdg01_1) # 2.700 ms

in_wdg11_1 = init_tensor(Val(1), s);
in_wdg11_2 = init_tensor(Val(1), s);
out_wdg11 = init_tensor(Val(2), s);

@benchmark wedge_product!(out_wdg11, Val((1,1)), s, in_wdg11_1, in_wdg11_2) # 2.030 ms

in_wdgdd01_0 = init_tensor_d(Val(0), s);
in_wdgdd01_1 = init_tensor_d(Val(1), s);
out_wdgdd01 = init_tensor_d(Val(1), s);

@benchmark wedge_product_dd!(out_wdgdd01, Val((0,1)), s, in_wdgdd01_0, in_wdgdd01_1) # 449.603 ms


# Sharp/flat

in_sharp_dd = init_tensor(Val(1), s);

out_sharp_dd_X = init_tensor_d(Val(0), s);
out_sharp_dd_Y = init_tensor_d(Val(0), s);

@benchmark sharp_dd!(out_sharp_dd_X, out_sharp_dd_Y, s, in_sharp_dd) # 465.870 ms

in_flat_dp_X = init_tensor_d(Val(0), s);
in_flat_dp_Y = init_tensor_d(Val(0), s);

out_flat_dp = init_tensor(Val(1), s);

@benchmark flat_dp!(out_flat_dp, s, in_flat_dp_X, in_flat_dp_Y) # 360.023 ms

# Periodic 

in_periodic_v = init_tensor(Val(0), s);
out_periodic_v = init_tensor(Val(0), s);

@benchmark boundary_v_map!(out_periodic_v, s, in_periodic_v, (1, 1)) # 1.335 ms

### MATRICES ###

d0_mat = exterior_derivative(Val(0), s) # 4.092 ms
input = ones(nv(s));
output = zeros(ne(s));

@benchmark mul!(res, d0_mat, input)

hdg_0 = hodge_star(Val(0), s)
input = ones(nv(s));
output = ones(nv(s));

@benchmark mul!(output, hdg_0, input) # 1.884 ms
