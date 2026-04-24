function form_initial_dist(x, y, std = 0.05)
  dist = MvNormal([x, y], [std, std])
  u_0 = map(p -> pdf(dist, SVector(p[1], p[2])), dual_points(s))
  u_0 .= u_0 .- minimum(u_0) # Shift to make minimum zero
  u_0 .= u_0 ./ maximum(u_0) # Scale to make maximum one
  return u_0
end

function form_initial_square(x, y, width = 0.2)
  half_width = width / 2
  return map(p -> (abs(p[1] - x) <= half_width && abs(p[2] - y) <= half_width) ? 1.0 : 0.0, dual_points(s))
end

const form_initial = form_initial_square

const dps = to_device(dual_points(s))

if simname == "Diagonal" # 3.1: Diagonal Velocity Field
  const phi_0 = form_initial(0.25, 0.25)

  # TODO: Change these to actual kernels for GPU support
  function generate_X(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1] + 1; z = p[2] + 1
      return 1/r * sin((2pi*t) / T)
    end
  end

  function generate_Y(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1] + 1; z = p[2] + 1
      return 1/r * sin((2pi*t) / T)
    end
  end
elseif simname == "Stretch" # 3.2: Stretching Velocity Field
  const phi_0 = form_initial(0.7, 0.25)

  function generate_X(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1]; z = p[2]
      return -r / 10 * sin((2pi*t) / T)
    end
  end

  function generate_Y(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1]; z = p[2]
      return z / 5 * sin((2pi*t) / T)
    end
  end
elseif simname == "Rotate" # 3.3: Rotating Velocity Field
  const phi_0 = form_initial(0.75, 0.25)

  function generate_X(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1] + 1; z = p[2] + 1
      return -2 * z / (pi * r) * sin((2pi*t) / T)
    end
  end

  function generate_Y(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1] + 1; z = p[2] + 1
      return 2/pi * sin((2pi*t) / T)
    end
  end
elseif simname == "CircularVortex" # 3.4: Circular Vortex
  const phi_0 = form_initial(0.4, 0.4)

  function generate_X(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1]; z = p[2]
      return -sin(pi * r)*cos(pi * z) * cos(2pi*t / T)
    end
  end

  function generate_Y(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1]; z = p[2]
      return 1/(pi*r) * sin(pi*z)*(sin(pi*r)+pi*r*cos(pi*r)) * cos(2pi*t / T)
    end
  end
elseif simname == "ReversedVortex" # 3.5: Reversed Single Vortex
  const phi_0 = form_initial(0.5, 0.75)

  function generate_X(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1]; z = p[2]
      return 1/r * 2*sin(pi * z) * cos(pi * z) * sin(pi * r)^2 * cos(2pi*t / T)
    end
  end

  function generate_Y(s::UniformCubicalComplex2D, t::Real, T::Real)
    map(dps) do p
      r = p[1]; z = p[2]
      return -1/r * 2*sin(pi * r) * cos(pi * r) * sin(pi * z)^2 * cos(2pi*t / T)
    end
  end
else
  error("Invalid case")
end

const phi_star_0  = to_device(cpu_inv_hdg_2 * phi_0)   # star representation of tracer
