using Test
using CairoMakie
using LinearAlgebra

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

lx = ly = 2π;
s = UniformCubicalComplex2D(201, 201, lx, ly);

function form_taylor_vortices(s::UniformCubicalComplex2D, G::Real, a::Real, centers)
  u = zeros(nv(s))

  function taylor_vortex(pnt::Point3d, cntr::Point3d)
    r = norm(pnt .- cntr)
    (G / a) * (2 - (r / a)^2) * exp(0.5 * (1 - (r / a)^2))
  end

  for center in centers
    u += map(p -> taylor_vortex(p, center), points(s))
  end

  return u
end 

ω = form_taylor_vortices(s, 1.0, 0.3, [Point3d(lx / 2 - 0.4, ly / 2, 0.0), Point3d(lx / 2 + 0.4, ly / 2, 0.0)])

save("imgs/NS/Vortices.png", plot_zeroform(s, ω))

# DEC Operators 
Δ0 = laplacian(Val(0), s);

d0 = exterior_derivative(Val(0), s);
d1 = exterior_derivative(Val(1), s);

dual_d0 = dual_derivative(Val(0), s);
dual_d1 = dual_derivative(Val(1), s);

d_beta = 0.5 * abs.(dual_d1) * spdiagm(dual_d0 * ones(nquads(s)));

hdg_1 = hodge_star(Val(1), s);

inv_hdg_0 = inv_hodge_star(Val(0), s);
inv_hdg_1 = inv_hodge_star(Val(1), s);

δ1 = codifferential(Val(1), s);

# Solve for initial conditions
ψ = Δ0 \ ω
ψ .= ψ .- minimum(ψ)
u_star_0 = d0 * ψ

ω_test = δ1 * u_star_0

save("imgs/NS/RoundTrip_Vortices.png", plot_zeroform(s, ω_test))

Δt = 1e-3
μ = 1e-3

rhs_U_mat = (-1 / Δt) * I + μ * d0 * inv_hdg_0 * dual_d1 * hdg_1
rhs_Pd_mat = inv_hdg_1 * dual_d0

Wv(v) = 0.5 * spdiagm(v)

vΔ1 = -μ * d0 * inv_hdg_0 * d_beta
adv_u_star = abs.(d0) * inv_hdg_0 * dual_d1 * hdg_1
adv_v = abs.(d0) * inv_hdg_0 * d_beta

X = zeros(nquads(s))
Y = zeros(nquads(s))

v = zeros(ne(s))

function generate_F(u_star)
    u = hdg_1 * u_star
    sharp_dd!(X, Y, s, u)
    flat_dp!(v, s, X, Y)

    # Diffusion and advection
    return (-1 / Δt) * u_star + vΔ1 * v + Wv(v) * (adv_u_star * u_star + adv_v * v)
end

# Generate matrix
rhs_top = hcat(rhs_U_mat, rhs_Pd_mat)
rhs_bottom = hcat(d1, spdiagm(zeros(nquads(s))))
rhs = vcat(rhs_top, rhs_bottom)

f_rhs = factorize(rhs)

U = zeros(ne(s) + nquads(s))

# TODO: Why is the end time so long, Mohamed paper sees behavior at t=5
tₑ = 5.0
steps = ceil(Int64, tₑ / Δt)
Us = []

u_star = deepcopy(u_star_0)

F₂ = zeros(nquads(s))
for (step, time) in enumerate(range(0, tₑ; step = Δt))
    F = vcat(generate_F(u_star), F₂)
    U .= f_rhs \ F
    u_star .= U[1:ne(s)]

    if (step - 1) % 50 == 0
        println("Loading simulation results at time $time: $(time / tₑ * 100)%")
        push!(Us, deepcopy(u_star))
    end
end

ω_end = δ1 * Us[end]

fig = plot_zeroform(s, ω_end) 
save("imgs/NS/FinalVortices.png", fig)

fig = plot_oneform(s, hdg_1 * Us[end])
save("imgs/NS/FinalVelocity.png", fig)

fig = plot_xy_oneform(s, hdg_1 * Us[end])
save("imgs/NS/FinalVelocityXY.png", fig)

# ωs = map(u -> δ1 * u, Us)

# function create_gif(solution, file_name)
#   frames = length(solution)
#   fig = Figure()
#   ax = CairoMakie.Axis(fig[1,1])
#   msh = CairoMakie.mesh!(ax, s, color=first(solution), colormap=:jet, colorrange=extrema(first(solution)))
#   Colorbar(fig[1,2], msh)
#   CairoMakie.record(fig, file_name, 1:frames; framerate = 15) do t
#     msh.color = solution[t]
#   end
# end

# create_gif(ωs, "imgs/NS/TaylorVortices.mp4")