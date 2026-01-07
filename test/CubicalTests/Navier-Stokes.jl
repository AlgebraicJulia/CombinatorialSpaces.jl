using Test
using CairoMakie
using LinearAlgebra

include("../../src/CubicalComplexes.jl")

lx = ly = 2π;
s = uniform_grid(lx, ly, 81, 81);

function form_taylor_vortices(s::HasCubicalComplex, G::Real, a::Real, centers)
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

save("imgs/Vortices.png", plot_zeroform(s, ω))

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

save("imgs/RoundTrip_Vortices.png", plot_zeroform(s, ω_test))

Δt = 1e-3
μ = 0

rhs_U_mat = (-1 / Δt) * I + μ * d0 * inv_hdg_0 * dual_d1 * hdg_1
rhs_Pd_mat = inv_hdg_1 * dual_d0

Wv(v) = spdiagm(v)

vΔ1 = -μ * d0 * inv_hdg_0 * d_beta
adv_u_star = 0.5 * abs.(d0) * inv_hdg_0 * dual_d1 * hdg_1
adv_v = 0.5 * abs.(d0) * inv_hdg_0 * d_beta

X = init_tensor_d(Val(0), s)
Y = init_tensor_d(Val(0), s)

v_ten = init_tensor(Val(1), s)

# TODO: Increase performance of this step (mainly in sharp)
function generate_F(u_star)
    u = hdg_1 * u_star
    sharp_dd!(X, Y, s, tensorfy(s, u))
    flat_dp!(v_ten, s, X, Y)

    v = detensorfy(Val(1), s, v_ten)

    # Diffusion and advection
    return (-1 / Δt) * u_star + vΔ1 * v + Wv(v) * (adv_u_star * u_star + adv_v * v)
end

# Generate matrix
rhs_top = hcat(rhs_U_mat, rhs_Pd_mat)
rhs_bottom = hcat(d1, spdiagm(zeros(nquads(s))))
rhs = vcat(rhs_top, rhs_bottom)

f_rhs = factorize(rhs)

U = zeros(ne(s) + nquads(s))

tₑ = 8.0
steps = ceil(Int64, tₑ / Δt)
Us = [u_star_0]

u_star = deepcopy(u_star_0)

F₂ = zeros(nquads(s))
for step in 1:steps
    F = vcat(generate_F(u_star), F₂)
    U .= f_rhs \ F
    u_star .= U[1:ne(s)]

    if step % 50 == 0
        println("Loading simulation results: $(step / steps * 100)%")
        push!(Us, deepcopy(u_star))
    end
end

ω_end = δ1 * u_star

fig = Figure();
ax = CairoMakie.Axis(fig[1, 1]) 
msh = CairoMakie.mesh!(ax, s, color=ω_end, colormap=:jet)
Colorbar(fig[1, 2], msh)
save("imgs/FinalVortices.png", fig)

ωs = map(u -> δ1 * u, Us)

function create_gif(solution, file_name)
  frames = length(solution)
  fig = Figure()
  ax = CairoMakie.Axis(fig[1,1])
  msh = CairoMakie.mesh!(ax, s, color=first(solution), colormap=:jet, colorrange=extrema(first(solution)))
  Colorbar(fig[1,2], msh)
  CairoMakie.record(fig, file_name, 1:frames; framerate = 15) do t
    msh.color = solution[t]
  end
end

create_gif(ωs, "imgs/TaylorVortices.mp4")