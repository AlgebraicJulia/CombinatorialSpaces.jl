const dps = interior(Val(2), dual_points(s), s)
const xs = map(a -> a[1], dps)
const ys = map(a -> a[2], dps)

function require_state_field(state, field::Symbol, func_name::AbstractString)
  hasproperty(state, field) || throw(ArgumentError("$(func_name) requires state.$(field), but it was not found"))
  return getproperty(state, field)
end

function plot_vorticity(s, state, file_end::String, time::String)
  U = require_state_field(state, :U, "plot_vorticity")
  ω = inv_hodge_star(Val(0), s) * dual_derivative(Val(1), s) * U

  fig = plot_zeroform(
    s,
    ω;
    axis_kwargs = (title = "Vorticity Field (ω) at $(time)",),
    mesh_kwargs = (colormap = :jet,),
  )
  save(joinpath(save_path, "Vorticity_$(file_end).png"), fig)
  return fig
end

function plot_density(s, state, file_end::String, time::String)
  rho = require_state_field(state, :rho, "plot_density")
  fig = plot_twoform(
    s,
    rho;
    axis_kwargs = (title = "Density Field (ρ) at $(time)",),
    heatmap_kwargs = (colormap = Reverse(:oslo),),
  )
  save(joinpath(save_path, "Density_$(file_end).png"), fig)
  return fig
end

function plot_momentum_magnitude(s, state, file_end::String, time::String)
  U = require_state_field(state, :U, "plot_momentum_magnitude")
  u = U

  X, Y = sharp_dd(s, u)
  fig = plot_twoform(
    s,
    sqrt.(X.^2 + Y.^2);
    axis_kwargs = (title = "Momentum Magnitude at $(time)",),
    heatmap_kwargs = (colormap = :viridis,),
  )
  save(joinpath(save_path, "Momentum_Magnitude_$(file_end).png"), fig)
  return fig
end

function plot_momentum_components(s, state, file_end::String, time::String)
  U = require_state_field(state, :U, "plot_momentum_components")
  u = U
  fig = plot_xy_oneform(
    s,
    u;
    axis_x_kwargs = (title = "X Component at $(time)", xlabel = "x", ylabel = "y"),
    axis_y_kwargs = (title = "Y Component at $(time)", xlabel = "x", ylabel = "y"),
    heatmap_x_kwargs = (colormap = :jet,),
    heatmap_y_kwargs = (colormap = :jet,),
  )

  save(joinpath(save_path, "Momentum_Components_$(file_end).png"), fig)
  return fig
end

function plot_pressure(s, state, file_end::String, time::String)
  Theta = require_state_field(state, :Theta, "plot_pressure")
  p = pressure(Theta)

  fig = plot_twoform(
    s,
    p;
    axis_kwargs = (title = "Pressure Field (p) at $(time)",),
    heatmap_kwargs = (colormap = :magma,),
  )
  save(joinpath(save_path, "Pressure_$(file_end).png"), fig)
  return fig
end

function create_mp4(::LMNSModel, filename::String, regular_states::AbstractVector; frames::Int = length(regular_states), framerate::Int = 15, records::Int = 50)
    isempty(regular_states) && return nothing

    jump = max(1, Int(floor(frames / records)))

    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1], title = "Velocity Field (u)")
    ax2 = CairoMakie.Axis(fig[1, 2], title = "Velocity Field (v)")
    ax3 = CairoMakie.Axis(fig[2, 1], title = "Density Field (ρ)")
    ax4 = CairoMakie.Axis(fig[2, 2], title = "Velocity Magnitude")

    step = Observable(1)

    X = Observable(zeros(nquadsr(s)))
    Y = Observable(zeros(nquadsr(s)))

    @lift((X[], Y[]) = interior.(Ref(Val(2)), sharp_dd(s, regular_states[$step].U), Ref(s)))
    rho = @lift(interior(Val(2), regular_states[$step].rho, s))
    mag = @lift(sqrt.($X.^2 + $Y.^2))

    heat1 = heatmap!(ax, xs, ys, X, colormap=:jet)
    heat2 = heatmap!(ax2, xs, ys, Y, colormap=:jet)
    heat3 = heatmap!(ax3, xs, ys, rho, colormap=Reverse(:oslo))
    heat4 = heatmap!(ax4, xs, ys, mag, colormap=:viridis)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))
    resize_to_layout!(fig)

    CairoMakie.record(fig, filename, 1:jump:frames; framerate = framerate) do i
        step[] = i
        notify(X); notify(Y); notify(rho); notify(mag)
    end

    return nothing
end

function create_mp4(::LMNSModel, regular_states::AbstractVector, file_end::String; frames::Int = length(regular_states), framerate::Int = 15, records::Int = 50)
  return create_mp4(
    LMNSModel(),
    joinpath(save_path, "simulation_$(file_end).mp4"),
    regular_states;
    frames = frames,
    framerate = framerate,
    records = records,
  )
end

function create_mp4(::LMNSHModel, filename::String, regular_states::AbstractVector; frames::Int = length(regular_states), framerate::Int = 15, records::Int = 50)
    isempty(regular_states) && return nothing

    jump = max(1, Int(floor(frames / records)))

    fig = Figure()
    ax1 = CairoMakie.Axis(fig[1, 1], title = "Velocity Field (u)")
    ax2 = CairoMakie.Axis(fig[1, 2], title = "Velocity Field (v)")
    ax3 = CairoMakie.Axis(fig[2, 1], title = "Density Field (ρ)")
    # ax4 = CairoMakie.Axis(fig[2, 2], title = "Velocity Magnitude")
    ax4 = CairoMakie.Axis(fig[2, 2], title = "Dc Potential Temperature (Θ)")
    ax5 = CairoMakie.Axis(fig[1, 3], title = "Potential Temperature (Θ)")
    # ax6 = CairoMakie.Axis(fig[2, 3], title = "Vorticity Field (ω)")
    ax6 = CairoMakie.Axis(fig[2, 3], title = "Pressure Field (p)")

    step = Observable(1)

    X = Observable(zeros(nquadsr(s)))
    Y = Observable(zeros(nquadsr(s)))

    @lift((X[], Y[]) = interior.(Ref(Val(2)), sharp_dd(s, regular_states[$step].U), Ref(s)))
    rho = @lift(interior(Val(2), regular_states[$step].rho, s))
    theta = @lift(interior(Val(2), regular_states[$step].Theta, s) ./ interior(Val(2), regular_states[$step].rho, s))
    Theta = @lift(interior(Val(2), regular_states[$step].Theta, s))
    # mag = @lift(sqrt.($X.^2 + $Y.^2))
    # vort = @lift(plot_inv_hdg_0 * plot_dual_d1 * Us[$step])
    p = @lift(pressure(interior(Val(2), regular_states[$step].Theta, s)))

    heatmap!(ax1, xs, ys, X, colormap=:jet)
    heatmap!(ax2, xs, ys, Y, colormap=:jet)
    heatmap!(ax3, xs, ys, rho, colormap=Reverse(:oslo))
    # heatmap!(ax4, xs, ys, mag, colormap=:viridis)
    heatmap!(ax4, xs, ys, Theta, colormap=:viridis)
    heatmap!(ax5, xs, ys, theta, colormap=:plasma)
    # mesh!(ax6, s; color = vort, colormap=:magma)
    heatmap!(ax6, xs, ys, p, colormap=:magma)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))
    colsize!(fig.layout, 3, Aspect(1, 1.0))
    resize_to_layout!(fig)

    CairoMakie.record(fig, filename, 1:jump:frames; framerate = framerate) do i
        step[] = i
        notify(X); notify(Y); notify(p); notify(theta); notify(Theta); notify(p) # notify(vort)
    end

    return nothing
end

function create_mp4(::LMNSHModel, regular_states::AbstractVector, file_end::String; frames::Int = length(regular_states), framerate::Int = 15, records::Int = 50)
  return create_mp4(
    LMNSHModel(),
    joinpath(save_path, "simulation_$(file_end).mp4"),
    regular_states;
    frames = frames,
    framerate = framerate,
    records = records,
  )
end

function create_mp4(model::AbstractSimulationModel, regular_save_values::SavedValues, file_end::String; frames::Int = length(regular_save_values.saveval), framerate::Int = 15, records::Int = 50)
    return create_mp4(model, regular_save_values.saveval, file_end; frames = frames, framerate = framerate, records = records)
end

function create_mp4(model::Union{LMNSModel, LMNSHModel}, filename::String, regular_save_values::SavedValues; frames::Int = length(regular_save_values.saveval), framerate::Int = 15, records::Int = 50)
  return create_mp4(model, filename, regular_save_values.saveval; frames = frames, framerate = framerate, records = records)
end

function mhd_velocity_magnitude(U::AbstractVector{Float64}, rho::AbstractVector{Float64})
  u = wedge_product_dd(Val(0), Val(1), s, 1 ./ rho, U)
  X, Y = sharp_dd(s, u)
  return sqrt.(X .^ 2 .+ Y .^ 2)
end

function mhd_magnetic_magnitude(B::AbstractVector{Float64})
  BX, BY = sharp_dd(s, B)
  return sqrt.(BX .^ 2 .+ BY .^ 2)
end

function mhd_density_colorrange()
  hasproperty(@__MODULE__, :rho_base) || return nothing
  hasproperty(@__MODULE__, :rho_perturbation) || return nothing

  rho0 = getproperty(@__MODULE__, :rho_base)
  drho = getproperty(@__MODULE__, :rho_perturbation)
  return (rho0 - drho, rho0 + drho)
end

mhd_current_density(B::AbstractVector{Float64}) = (dec_ops.hdg_2 * dec_ops.d1 * (-dec_ops.inv_hdg_1 * B)) ./ μ₀

function plot_mhd_vorticity(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  ω = dec_ops.inv_hdg_0 * dec_ops.dual_d1 * state.U
  save(joinpath(save_path, "Vorticity$(suffix_txt).png"), plot_zeroform(s, ω))
  return nothing
end

function plot_mhd_density(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  colorrange = mhd_density_colorrange()
  fig = isnothing(colorrange) ?
    plot_twoform(s, state.rho) :
    plot_twoform(s, state.rho; heatmap_kwargs = (colorrange = colorrange,))
  save(joinpath(save_path, "Density$(suffix_txt).png"), fig)
  return nothing
end

function plot_mhd_velocity_magnitude(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  vel_mag = mhd_velocity_magnitude(state.U, state.rho)
  save(joinpath(save_path, "VelocityMagnitude$(suffix_txt).png"), plot_twoform(s, vel_mag))
  return nothing
end

function plot_mhd_magnetic_magnitude(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  B_mag = mhd_magnetic_magnitude(state.B)
  save(joinpath(save_path, "MagneticFieldMagnitude$(suffix_txt).png"), plot_twoform(s, B_mag))
  return nothing
end

function plot_mhd_velocity_components(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  save(joinpath(save_path, "VelocityComponents$(suffix_txt).png"), plot_xy_oneform(s, state.U))
  return nothing
end

function plot_mhd_magnetic_components(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  save(joinpath(save_path, "MagneticFieldComponents$(suffix_txt).png"), plot_xy_oneform(s, state.B))
  return nothing
end

function plot_mhd_current_density(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  Jz = mhd_current_density(state.B)
  save(joinpath(save_path, "CurrentDensity$(suffix_txt).png"), plot_twoform(s, Jz))
  return nothing
end

function plot_mhd_momentum_divergence(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  U_div = dec_ops.inv_hdg_0 * dec_ops.dual_d1 * state.U
  save(joinpath(save_path, "MomentumDivergence$(suffix_txt).png"), plot_zeroform(s, U_div))
  return nothing
end

function save_mhd_diagnostics(state::NamedTuple{(:U, :rho, :B), <:Tuple}; suffix::String = "")
  plot_mhd_vorticity(state; suffix = suffix)
  plot_mhd_density(state; suffix = suffix)
  plot_mhd_velocity_magnitude(state; suffix = suffix)
  plot_mhd_magnetic_magnitude(state; suffix = suffix)
  plot_mhd_velocity_components(state; suffix = suffix)
  plot_mhd_magnetic_components(state; suffix = suffix)
  plot_mhd_current_density(state; suffix = suffix)
  plot_mhd_momentum_divergence(state; suffix = suffix)
  return nothing
end

function save_mhd_diagnostics(regular_save_values::SavedValues; suffix::String = "", records::Int = 200, framerate::Int = 15)
  isempty(regular_save_values.saveval) && return nothing

  state_end = regular_save_values.saveval[end]
  save_mhd_diagnostics(state_end; suffix = suffix)

  suffix_txt = isempty(suffix) ? "" : "_$(suffix)"
  create_mp4(
    MHDModel(),
    joinpath(save_path, "$(simname)$(suffix_txt).mp4"),
    regular_save_values;
    framerate = framerate,
    records = records,
  )

  return nothing
end

function create_mp4(::MHDModel, filename::String, states::AbstractVector{<:NamedTuple}; frames::Int = length(states), framerate::Int = 15, records::Int = 200)
  nqx = nxquads(s) - 2 * hx(s)
  nqy = nyquads(s) - 2 * hy(s)

  dps = interior(Val(2), dual_points(s), s)
  unique_x = sort(unique(map(a -> a[1], dps)))
  unique_y = sort(unique(map(a -> a[2], dps)))

  fig = Figure(size = (1200, 800))
  ax1 = CairoMakie.Axis(fig[1, 1], title = "Density")
  ax2 = CairoMakie.Axis(fig[1, 3], title = "Magnetic Field Magnitude")
  ax3 = CairoMakie.Axis(fig[2, 1], title = "Velocity Magnitude")
  ax4 = CairoMakie.Axis(fig[2, 3], title = "Current Density (Jz)")

  step = Observable(1)

  rho = @lift(reshape(interior(Val(2), states[$step].rho, s), nqx, nqy))
  B_mag = @lift(begin
    B_mag_step = mhd_magnetic_magnitude(states[$step].B)
    reshape(interior(Val(2), B_mag_step, s), nqx, nqy)
  end)
  vel_mag = @lift(begin
    vel_mag_step = mhd_velocity_magnitude(states[$step].U, states[$step].rho)
    reshape(interior(Val(2), vel_mag_step, s), nqx, nqy)
  end)
  Jz = @lift(reshape(interior(Val(2), mhd_current_density(states[$step].B), s), nqx, nqy))

  rho_colorrange = mhd_density_colorrange()
  rho_heatmap_kwargs = isnothing(rho_colorrange) ?
    (colormap = Reverse(:oslo),) :
    (colormap = Reverse(:oslo), colorrange = rho_colorrange)

  h1 = heatmap!(ax1, unique_x, unique_y, rho; rho_heatmap_kwargs...)
  Colorbar(fig[1, 2], h1)

  h2 = heatmap!(ax2, unique_x, unique_y, B_mag, colormap = :inferno)
  Colorbar(fig[1, 4], h2)

  h3 = heatmap!(ax3, unique_x, unique_y, vel_mag, colormap = :viridis)
  Colorbar(fig[2, 2], h3)

  h4 = heatmap!(ax4, unique_x, unique_y, Jz, colormap = Reverse(:acton))
  Colorbar(fig[2, 4], h4)

  colsize!(fig.layout, 1, Aspect(1, 1.0))
  colsize!(fig.layout, 3, Aspect(1, 1.0))
  resize_to_layout!(fig)

  jump = max(1, Int(floor(frames / records)))
  CairoMakie.record(fig, filename, 1:jump:frames; framerate = framerate) do i
    step[] = i
  end

  return nothing
end

function create_mp4(::MHDModel, filename::String, regular_save_values::SavedValues; frames::Int = length(regular_save_values.saveval), framerate::Int = 15, records::Int = 200)
  create_mp4(MHDModel(), filename, regular_save_values.saveval; frames = frames, framerate = framerate, records = records)
end
