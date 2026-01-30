# function lr_boundary_verts_map!(v, s::HasCubicalComplex, depth::Int)
#   for y in 1:ny(s)
#     for x in 1:depth # Map right data to left boundary
#       v[coord_to_vert(s, x, y)] = v[coord_to_vert(s, nx(s) - 2 * depth + x, y)]
#     end
#     for x in 1:depth # Map left data to right boundary
#       v[coord_to_vert(s, nx(s) - x + 1, y)] = v[coord_to_vert(s, 2 * depth - x + 1, y)]
#     end
#   end

#   return v
# end

# function tb_boundary_verts_map!(v, s::HasCubicalComplex, depth::Int)
#   for x in 1:nx(s)
#     for y in 1:depth # Map top data to bottom boundary
#       v[coord_to_vert(s, x, y)] = v[coord_to_vert(s, x, ny(s) - 2 * depth + y)]
#     end
#     for y in 1:depth # Map bottom data to top boundary
#       v[coord_to_vert(s, x, ny(s) - y + 1)] = v[coord_to_vert(s, x, 2 * depth - y + 1)]
#     end
#   end

#   return v
# end

# function lr_boundary_quads_map!(v, s::HasCubicalComplex, depth::Int)
#   for y in 1:nyquads(s)
#     for x in 1:depth # Map right data to left boundary
#       v[coord_to_vert(s, x, y)] = v[coord_to_vert(s, nxquads(s) - 2 * depth + x, y)]
#     end
#     for x in 1:depth # Map left data to right boundary
#       v[coord_to_vert(s, nxquads(s) - x + 1, y)] = v[coord_to_vert(s, 2 * depth - x + 1, y)]
#     end
#   end

#   return v
# end

# function tb_boundary_quads_map!(v, s::HasCubicalComplex, depth::Int)
#   for x in 1:nxquads(s)
#     for y in 1:depth # Map top data to bottom boundary
#       v[coord_to_quad(s, x, y)] = v[coord_to_quad(s, x, nyquads(s) - 2 * depth + y)]
#     end
#     for y in 1:depth # Map bottom data to top boundary
#       v[coord_to_quad(s, x, nyquads(s) - y + 1)] = v[coord_to_quad(s, x, 2 * depth - y + 1)]
#     end
#   end

#   return v
# end

function boundary_v_map!(res, s::HasCubicalComplex, f, depths::Tuple)
  backend = get_backend(f)

  kernel = boundary_map_kernel(backend, 32, size(res))
  kernel(res, s, f, depths, ndrange = size(res))
  return res
end

function boundary_e_map!(res, s::HasCubicalComplex, f, depths::Tuple)
  backend = get_backend(xedges(f))

  for (res_set, f_set, depth) in zip(res, f, depths)
    kernel = boundary_map_kernel(backend, 32, size(res_set))
    kernel(res_set, s, f_set, depth, ndrange = size(res_set))
  end
  return res
end

function boundary_quad_map!(res, s::HasCubicalComplex, f, depths::Tuple)
  backend = get_backend(f)

  kernel = boundary_map_kernel(backend, 32, size(res))
  kernel(res, s, f, depths, ndrange = size(res))
  return res
end

@kernel function boundary_map_kernel(res, s::EmbeddedCubicalComplex2D, @Const(f), depths::Tuple{Int, Int})
  idx = @index(Global, Cartesian)
  x, y = idx.I

  xdepth, ydepth = depths

  res[idx] = f[idx]

  n_x, n_y = size(res)

  @inbounds if x <= xdepth # Right to left
    res[idx] = f[n_x - 2 * xdepth + x, y]
  elseif x >= n_x - xdepth + 1 # Left to right
    res[idx] = f[2 * xdepth + x - n_x, y]
  end

  @inbounds if y <= ydepth # Top to bottom
    res[idx] = f[x, n_y - 2 * ydepth + y]
  elseif y >= n_y - ydepth + 1 # Bottom to top
    res[idx] = f[x, 2 * ydepth + y - n_y]
  end
end
