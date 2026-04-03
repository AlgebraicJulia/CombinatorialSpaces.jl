using Test
using CairoMakie
using LinearAlgebra

include("../../src/CubicalCode/UniformMesh.jl")
include("../../src/CubicalCode/UniformMatrixDEC.jl")
include("../../src/CubicalCode/UniformPlotting.jl")

# edge_len(s, X_ALIGN) # CFL Condition
testcase = "RE1000"

if testcase == "RE10"
  Δt = 5e-3
  tₑ = 1.0
  Re = 10.0
  _nx = _ny = 129
elseif testcase == "RE100"
  Δt = 5e-3
  tₑ = 5.0
  Re = 100.0
  _nx = _ny = 129
elseif testcase == "RE1000"
  Δt = 5e-3
  tₑ = 25.0
  Re = 1_000.0
  _nx = _ny = 129
elseif testcase == "RE10000"
  Δt = 1e-4
  tₑ = 5.0
  Re = 10_000.0
  _nx = _ny = 257
end

lx = ly = 1.0;
s = UniformCubicalComplex2D(_nx, _ny, lx, ly);

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

μ = 1 / Re

# Boundary edges
top = top_edges(s);
bottom = bottom_edges(s);
left = left_edges(s);
right = right_edges(s);

boundary_idxs = findall(x -> x != 0, dual_d0 * ones(nquads(s)));
interior_idxs = findall(x -> x == 0, dual_d0 * ones(nquads(s)));

# Matrix components
function generate_rhs_U_mat()
    topmat = (-1 / Δt) * I + μ * d0 * inv_hdg_0 * dual_d1 * hdg_1

    botmat = deepcopy(d1)

    mat = vcat(topmat, botmat)
    mat[boundary_idxs, :] .= 0.0
    for i in boundary_idxs
        mat[i, i] = 1.0 # Enforce zero velocity on boundary for n+1 U values
    end
    return mat
end

rhs_U_mat = generate_rhs_U_mat()

function generate_rhs_P_mat()
    topmat = inv_hdg_1 * dual_d0
    topmat[boundary_idxs, :] .= 0.0 # No pressure gradient across boundary for n+1 P values

    botmat = spzeros(nquads(s), nquads(s))

    mat = vcat(topmat, botmat)
    return mat
end

rhs_P_mat = generate_rhs_P_mat()

fix_p_mat = sparse([1], [ne(s) + 1], [1.0], 1, ne(s) + nquads(s))
# Generate matrix
rhs = vcat(hcat(rhs_U_mat, rhs_P_mat), fix_p_mat)
f_rhs = factorize(rhs)

vΔ1 = -μ * d0 * inv_hdg_0 * d_beta
adv_u_star = abs.(d0) * inv_hdg_0 * dual_d1 * hdg_1
adv_v = abs.(d0) * inv_hdg_0 * d_beta

X = zeros(nquads(s))
Y = zeros(nquads(s))

v = zeros(ne(s))

Wv(v) = 0.5 * spdiagm(v)

function generate_F(u_star)
    u_star[boundary_idxs] .= 0.0 # Enforce zero velocity on boundary for current step (n) U values

    u = hdg_1 * u_star
    sharp_dd!(X, Y, s, u)
    flat_dp!(v, s, X, Y)

    v[boundary_idxs] .= 0.0
    v[top] .= edge_len(s, X_ALIGN)

    # Diffusion and advection
    src = (-1 / Δt) * u_star + vΔ1 * v + Wv(v) * (adv_u_star * u_star + adv_v * v)
    src[boundary_idxs] .= 0.0 # Enforce zero velocity on boundary for source term
    return src
end

# Functions for turbulent moving lid analysis
function generate_perturbation()
    u = zeros(ne(s))
    uten = tensorfy(s, deepcopy(u))
    fh, fv = uten

    # Applying perturbation u formula to fh entries
    (xh, yh) = size(fh)
    for i in 1:xh
        for j in 1:yh
            fh[i, j] = 2 * π * 10^-5 * cos(2 * π * (j - 1) / (yh - 1)) * sin(2 * π * (i - 1) / (xh - 1))
        end
    end

    # Applying perturbation v formula to fv entries
    (xv, yv) = size(fv)
    for i in 1:xv
        for j in 1:yv
            fv[i, j] = -2 * π * 10^-5 * cos(2 * π * (i - 1) / (xv - 1)) * sin(2 * π * (j - 1) / (yv - 1))
        end
    end

    u_star_0 = detensorfy(Val(1), s, uten)

    return u_star_0
end

function random_perturbation()
    Random.seed!(1234)
    u_star_0 = 1e-5 * randn(ne(s))
    u_star_0[boundary_idxs] .= 0.0
    return u_star_0
end

function get_magnitude(U)
    X = zeros(nquads(s))
    Y = zeros(nquads(s))
    sharp_dd!(X, Y, s, U)
    return sqrt.(X.^2 + Y.^2)
end

function find_stream(u_star)
    ω = inv_hdg_0 * dual_d1 * hdg_1 * u_star
    ψ = Δ0 \ ω
    ψ = ψ .- minimum(ψ)
    return ψ
end

function new_u_star(string)
    df = CSV.read(string, DataFrame);
    u_last = df[:, end];
    u_last = -inv_hdg_1 * u_last;
    return u_last
end

# Simulation number
sim = 3;

# Creating initial conditions based on previous simulation results for turbulent moving lid analysis
u_star_0 = new_u_star("test/CubicalTests/Flow Data/Sim$(sim-1)/FinalVelocity_Sim2_Re=$(Re).csv");

# Creating initial plots of velocity magnitude, velocity field, and stream function
fig = plot_twoform(s, get_magnitude(u_star_0))
save("test/CubicalTests/imgs/Turbulent Plots/Sim$(sim)/InitialVelocityMagnitude_Re=$(Re).png", fig)
fig = plot_xy_oneform(s, u_star_0)
save("test/CubicalTests/imgs/Turbulent Plots/Sim$(sim)/InitialVelocityXY_Re=$(Re).png", fig)
ψ = find_stream(u_star_0)
fig = plot_zeroform(s, ψ)
save("test/CubicalTests/imgs/Turbulent Plots/Sim$(sim)/InitialStreamFunction_Re=$(Re).png", fig)

u_star_0 = zeros(ne(s))

function runsim(tₑ, Δt, u_star_0; save_every = 50)
    u_star = deepcopy(u_star_0)

    Us = Vector{Float64}[]; Ps = Vector{Float64}[];
    push!(Us, hdg_1 * u_star)
    push!(Ps, zeros(nquads(s)))

    # Zeros for continuity + U bcs (zero on boundary) + P bc (fix pressure at one point)
    F₂ = zeros(nquads(s)+1)

    U = zeros(ne(s) + nquads(s))
    for (step, time) in enumerate(range(0, tₑ; step = Δt))
        F = vcat(generate_F(u_star), F₂)
        U .= f_rhs \ F
        u_star .= U[1:ne(s)]

        if any(isnan.(U))
            error("NaN values encountered in solution at time $time")
        end

        if (step - 1) % save_every == 0
            println("Loading simulation results at time $time: $(time / tₑ * 100)%")
            push!(Us, deepcopy(hdg_1 * u_star))
            push!(Ps, deepcopy(U[ne(s)+1:end]))
        end
    end

    return Us, Ps
end

Us, Ps = runsim(tₑ, Δt, u_star_0; save_every = 10);

time = length(Us)

# Saving vector and pressure data in CSV format
# Saving all of the Us vectors into a CSV file for later analysis
M = hcat(Us...);
df = DataFrame(M, :auto)
rename!(df, ["U$(i)" for i in 1:ncol(df)])
CSV.write("test/CubicalTests/Flow Data/Sim$(sim)/Velocity_Sim$(sim)_Re=$(Re).csv", df)

# Saving all of the Ps vectors into a CSV file for later analysis
M = hcat(Ps...);
df = DataFrame(M, :auto)
rename!(df, ["P$(i)" for i in 1:ncol(df)])
CSV.write("test/CubicalTests/Flow Data/Sim$(sim)/Pressure_Sim$(sim)_Re=$(Re).csv", df)

# Creating vectors for Us[end] and Ps[end] to save into CSV
Us_final = Us[end];
Ps_final = Ps[end];

CSV.write("test/CubicalTests/Flow Data/Sim$(sim)/FinalVelocity_Sim$(sim)_Re=$(Re).csv", DataFrame(U = Us_final))
CSV.write("test/CubicalTests/Flow Data/Sim$(sim)/FinalPressure_Sim$(sim)_Re=$(Re).csv", DataFrame(P = Ps_final))



fig = plot_oneform(s, Us[time], lengthscale = 0.01, normalize = false)
save("imgs/CF/FinalVelocity_Re=$(Re).png", fig)

fig = plot_xy_oneform(s, Us[time])
save("imgs/CF/FinalVelocityXY_Re=$(Re).png", fig)

tmp = Ps[time];
# Clip pressure values for better visualization
fig = plot_twoform(s, tmp)
save("imgs/CF/FinalPressure_Re=$(Re).png", fig)

vort = inv_hdg_0 * dual_d1 * Us[time];
fig = plot_zeroform(s, vort) # Log scale for better visualization
save("imgs/CF/FinalVortices_Re=$(Re).png", fig)

ψ = Δ0 \ vort;
fig = plot_zeroform(s, ψ)
save("imgs/CF/FinalStreamfunction_Re=$(Re).png", fig)

# Plot divergence of velocity field to check incompressibility
divergence = d1 * inv_hdg_1 * Us[time];
fig = plot_twoform(s, divergence)
save("imgs/CF/FinalDivergence_Re=$(Re).png", fig)

function interpolate_velocity(V, U, s)
    X = zeros(nquads(s))
    Y = zeros(nquads(s))
    sharp_dd!(X, Y, s, Us[time])
    flat_dp!(V, s, X, Y)
    return V
end

fig = plot_twoform(s, get_magnitude(Us[time]))
save("imgs/CF/FinalVelocityMagnitude_Re=$(Re).png", fig)

function create_mp4(filename, Us, Ps; frames::Int = length(Us), framerate::Int = 15, records::Int = 50)

    jump = max(1, Int(floor(frames / records)))

    fig = Figure()
    ax = Axis(fig[1, 1], title = "Velocity Field (u)")
    ax2 = Axis(fig[1, 2], title = "Velocity Field (v)")
    ax3 = Axis(fig[2, 1], title = "Pressure Field (p)")
    ax4 = Axis(fig[2, 2], title = "Velocity Magnitude")

    dps = dual_points(s)
    xs = map(a -> a[1], dps)
    ys = map(a -> a[2], dps)

    step = Observable(1)

    X = Observable(zeros(nquads(s)))
    Y = Observable(zeros(nquads(s)))

    @lift(sharp_dd!(X[], Y[], s, Us[$step]))
    p = @lift(Ps[$step])
    mag = @lift(sqrt.($X.^2 + $Y.^2))

    heat1 = heatmap!(ax, xs, ys, X, colormap=:jet, colorrange=(-1, 1))
    heat2 = heatmap!(ax2, xs, ys, Y, colormap=:jet, colorrange=(-1, 1))
    heat3 = heatmap!(ax3, xs, ys, p, colormap=:jet)
    heat4 = heatmap!(ax4, xs, ys, mag, colormap=:jet)

    CairoMakie.record(fig, filename, 1:jump:frames; framerate = framerate) do i
        step[] = i
        notify(X); notify(Y)
    end
end

create_mp4("imgs/CF/simulation_Re=$(Re).mp4", Us, Ps)

# # Debugging

# # Noted high divergence at top right corner
# debug_time = length(Us)

# extrema(Us[debug_time][boundary_idxs])
# Ps[debug_time][1]

# all(0 .<= get_magnitude(Us[debug_time]) .<= 1.0) # Velocity magnitudes should be reasonable

# x = nxquads(s); y = nyquads(s);
# quad_idx = coord_to_quad(s, x, y);
# tmp = divergence[quad_idx];

# # Edges are ordered as: bottom, right, top, left
# e1, e2, e3, e4 = quad_edges(s, x, y);

# # Flux through primal edges
# U = inv_hdg_1 * Us[debug_time];
# u1 = U[e1]; u2 = U[e2]; u3 = U[e3]; u4 = U[e4];

# # Should be zero due to boundary condition
# u2 == 0.0
# u3 == 0.0

# # Should be close to zero, but is not, why?
# u1 - u4

# X_debug = zeros(nquads(s));
# Y_debug = zeros(nquads(s));
# sharp_dd!(X_debug, Y_debug, s, Us[debug_time]);
# V = zeros(ne(s));
# flat_dp!(V, s, X_debug, Y_debug);

# v1 = V[e1]; v2 = V[e2]; v3 = V[e3]; v4 = V[e4];


# P = Ps[debug_time];
# dP = rhs_Pd_mat * P;
# dP = inv_hdg_1 * dual_d0 * P;

# all(dP[boundary_idxs] .== 0.0) # Should be zero due to pressure bc on boundary

# dp1 = dP[e1]; dp2 = dP[e2]; dp3 = dP[e3]; dp4 = dP[e4]

# dp2 == 0.0
# dp3 == 0.0

# dp1
# dp4
