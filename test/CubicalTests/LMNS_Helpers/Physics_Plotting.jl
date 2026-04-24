const cpu_inv_hdg_0 = inv_hodge_star(Val(0), s)
const cpu_dual_d1 = dual_derivative(Val(1), s)
const cpu_hdg_1 = hodge_star(Val(1), s)

const dps = interior(Val(2), dual_points(s), s)
const xs = map(a -> a[1], dps)
const ys = map(a -> a[2], dps)

function plot_vorticity(s, U::AbstractVector, file_end::String, time::String)
  ω = cpu_inv_hdg_0 * cpu_dual_d1 * U;

  fig = Figure()
  ax = CairoMakie.Axis(fig[1, 1], title = "Vorticity Field (ω) at $(time)")
  msh = mesh!(ax, s; color = interior(Val(0), ω, s), colormap=:jet)
  Colorbar(fig[1, 2], msh)
  save(joinpath(save_path, "Vorticity_$(file_end).png"), fig)
  return fig
end

function plot_density(s, rho::AbstractVector, file_end::String, time::String)
  fig = Figure()
  ax = CairoMakie.Axis(fig[1, 1], title = "Density Field (ρ) at $(time)")
  hm = heatmap!(ax, xs, ys, interior(Val(2), rho, s); colormap=Reverse(:oslo))
  Colorbar(fig[1, 2], hm)
  save(joinpath(save_path, "Density_$(file_end).png"), fig)
  return fig
end

function plot_velocity_magnitude(s, U::AbstractVector, file_end::String, time::String)
  u = cpu_hdg_1 * U;
  X = zeros(nquads(s))
  Y = zeros(nquads(s))

  X, Y = sharp_dd(s, u)
  X = interior(Val(2), X, s)
  Y = interior(Val(2), Y, s)
  fig = Figure()
  ax = CairoMakie.Axis(fig[1, 1], title = "Velocity Magnitude at $(time)")
  hm = heatmap!(ax, xs, ys, sqrt.(X.^2 + Y.^2), colormap=:viridis)
  Colorbar(fig[1, 2], hm)
  save(joinpath(save_path, "VelocityMagnitude_$(file_end).png"), fig)
  return fig
end

function plot_velocity_components(s, U::AbstractVector, file_end::String, time::String)
  u = cpu_hdg_1 * U;
  X = zeros(nquads(s))
  Y = zeros(nquads(s))

  X, Y = sharp_dd(s, u)
  X = interior(Val(2), X, s)
  Y = interior(Val(2), Y, s)
  fig = Figure()
  axX = CairoMakie.Axis(fig[1,1]; title = "X Component at $(time)", xlabel = "x", ylabel = "y")
  hmX = heatmap!(axX, xs, ys, X, colormap=:jet)
  Colorbar(fig[1, 2], hmX)

  axY = CairoMakie.Axis(fig[2,1]; title = "Y Component at $(time)", xlabel = "x", ylabel = "y")
  hmY = heatmap!(axY, xs, ys, Y, colormap=:jet)
  Colorbar(fig[2, 2], hmY)

  save(joinpath(save_path, "VelocityComponents_$(file_end).png"), fig)
  return fig
end

function plot_pressure(s, Theta::AbstractVector, file_end::String, time::String)
  p = pressure(Theta)

  fig = Figure()
  ax = CairoMakie.Axis(fig[1, 1], title = "Pressure Field (p) at $(time)")
  hm = heatmap!(ax, xs, ys, interior(Val(2), p, s); colormap=:magma)
  Colorbar(fig[1, 2], hm)
  save(joinpath(save_path, "Pressure_$(file_end).png"), fig)
  return fig
end

# This is for Isothermal simulations where pressure is directly proportional to density
function create_mp4(Us, rhos, file_end::String; frames::Int = length(Us), framerate::Int = 15, records::Int = 50)

    jump = max(1, Int(floor(frames / records)))

    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1], title = "Velocity Field (u)")
    ax2 = CairoMakie.Axis(fig[1, 2], title = "Velocity Field (v)")
    ax3 = CairoMakie.Axis(fig[2, 1], title = "Density Field (ρ)")
    ax4 = CairoMakie.Axis(fig[2, 2], title = "Velocity Magnitude")

    step = Observable(1)

    X = Observable(zeros(nquadsr(s)))
    Y = Observable(zeros(nquadsr(s)))

    @lift((X[], Y[]) = interior.(Ref(Val(2)), sharp_dd(s, Us[$step]), Ref(s)))
    rho = @lift(interior(Val(2), rhos[$step], s))
    mag = @lift(sqrt.($X.^2 + $Y.^2))

    heat1 = heatmap!(ax, xs, ys, X, colormap=:jet)
    heat2 = heatmap!(ax2, xs, ys, Y, colormap=:jet)
    heat3 = heatmap!(ax3, xs, ys, rho, colormap=Reverse(:oslo))
    heat4 = heatmap!(ax4, xs, ys, mag, colormap=:viridis)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))
    resize_to_layout!(fig)

    CairoMakie.record(fig, joinpath(save_path, "simulation_$(file_end).mp4"), 1:jump:frames; framerate = framerate) do i
        step[] = i
        notify(X); notify(Y); notify(rho); notify(mag)
    end
end

function create_mp4(Us, rhos, Thetas, file_end::String; frames::Int = length(Us), framerate::Int = 15, records::Int = 50)

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

    @lift((X[], Y[]) = interior.(Ref(Val(2)), sharp_dd(s, Us[$step]), Ref(s)))
    rho = @lift(interior(Val(2), rhos[$step], s))
    theta = @lift(interior(Val(2), Thetas[$step], s) ./ interior(Val(2), rhos[$step], s))
    Theta = @lift(interior(Val(2), Thetas[$step], s))
    # mag = @lift(sqrt.($X.^2 + $Y.^2))
    # vort = @lift(cpu_inv_hdg_0 * cpu_dual_d1 * Us[$step])
    p = @lift(pressure(interior(Val(2), Thetas[$step], s)))

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

    CairoMakie.record(fig, joinpath(save_path, "simulation_$(file_end).mp4"), 1:jump:frames; framerate = framerate) do i
        step[] = i
        notify(X); notify(Y); notify(p); notify(theta); notify(Theta); notify(p) # notify(vort)
    end
end

  function create_mp4(regular_states::AbstractVector, file_end::String; frames::Int = length(regular_states), framerate::Int = 15, records::Int = 50)
    isempty(regular_states) && return nothing

    jump = max(1, Int(floor(frames / records)))

    fig = Figure()
    ax1 = CairoMakie.Axis(fig[1, 1], title = "Velocity Field (u)")
    ax2 = CairoMakie.Axis(fig[1, 2], title = "Velocity Field (v)")
    ax3 = CairoMakie.Axis(fig[2, 1], title = "Density Field (ρ)")
    ax4 = CairoMakie.Axis(fig[2, 2], title = "Dc Potential Temperature (Θ)")
    ax5 = CairoMakie.Axis(fig[1, 3], title = "Potential Temperature (Θ)")
    ax6 = CairoMakie.Axis(fig[2, 3], title = "Pressure Field (p)")

    step = Observable(1)

    X = Observable(zeros(nquadsr(s)))
    Y = Observable(zeros(nquadsr(s)))

    @lift((X[], Y[]) = interior.(Ref(Val(2)), sharp_dd(s, regular_states[$step].U), Ref(s)))
    rho = @lift(interior(Val(2), regular_states[$step].rho, s))
    theta = @lift(interior(Val(2), regular_states[$step].Theta, s) ./ interior(Val(2), regular_states[$step].rho, s))
    Theta = @lift(interior(Val(2), regular_states[$step].Theta, s))
    p = @lift(pressure(interior(Val(2), regular_states[$step].Theta, s)))

    heatmap!(ax1, xs, ys, X, colormap=:jet)
    heatmap!(ax2, xs, ys, Y, colormap=:jet)
    heatmap!(ax3, xs, ys, rho, colormap=Reverse(:oslo))
    heatmap!(ax4, xs, ys, Theta, colormap=:viridis)
    heatmap!(ax5, xs, ys, theta, colormap=:plasma)
    heatmap!(ax6, xs, ys, p, colormap=:magma)

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))
    colsize!(fig.layout, 3, Aspect(1, 1.0))
    resize_to_layout!(fig)

    CairoMakie.record(fig, joinpath(save_path, "simulation_$(file_end).mp4"), 1:jump:frames; framerate = framerate) do i
      step[] = i
      notify(X); notify(Y); notify(rho); notify(theta); notify(Theta); notify(p)
    end

    return nothing
  end

  function create_mp4(regular_save_values, file_end::String; frames::Int = length(regular_save_values.saveval), framerate::Int = 15, records::Int = 50)
    return create_mp4(regular_save_values.saveval, file_end; frames = frames, framerate = framerate, records = records)
  end
