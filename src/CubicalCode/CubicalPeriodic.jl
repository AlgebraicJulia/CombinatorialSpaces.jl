function lr_boundary_verts_map!(v, s::HasCubicalComplex, depth::Int)
  for y in 1:ny(s)
    for x in 1:depth # Map right data to left boundary
      v[coord_to_vert(s, x, y)] = v[coord_to_vert(s, nx(s) - 2 * depth + x, y)]
    end
    for x in 1:depth # Map left data to right boundary
      v[coord_to_vert(s, nx(s) - x + 1, y)] = v[coord_to_vert(s, 2 * depth - x + 1, y)]
    end
  end

  return v
end

function tb_boundary_verts_map!(v, s::HasCubicalComplex, depth::Int)
  for x in 1:nx(s)
    for y in 1:depth # Map top data to bottom boundary
      v[coord_to_vert(s, x, y)] = v[coord_to_vert(s, x, ny(s) - 2 * depth + y)]
    end
    for y in 1:depth # Map bottom data to top boundary
      v[coord_to_vert(s, x, ny(s) - y + 1)] = v[coord_to_vert(s, x, 2 * depth - y + 1)]
    end
  end

  return v
end

function lr_boundary_quads_map!(v, s::HasCubicalComplex, depth::Int)
  for y in 1:nyquads(s)
    for x in 1:depth # Map right data to left boundary
      v[coord_to_vert(s, x, y)] = v[coord_to_vert(s, nxquads(s) - 2 * depth + x, y)]
    end
    for x in 1:depth # Map left data to right boundary
      v[coord_to_vert(s, nxquads(s) - x + 1, y)] = v[coord_to_vert(s, 2 * depth - x + 1, y)]
    end
  end

  return v
end

function tb_boundary_quads_map!(v, s::HasCubicalComplex, depth::Int)
  for x in 1:nxquads(s)
    for y in 1:depth # Map top data to bottom boundary
      v[coord_to_quad(s, x, y)] = v[coord_to_quad(s, x, nyquads(s) - 2 * depth + y)]
    end
    for y in 1:depth # Map bottom data to top boundary
      v[coord_to_quad(s, x, nyquads(s) - y + 1)] = v[coord_to_quad(s, x, 2 * depth - y + 1)]
    end
  end

  return v
end

# TODO: Convert these to kernels
function lr_boundary_coord_verts_map!(v, s::HasCubicalComplex, depth::Int)
  for y in 1:ny(s)
    for x in 1:depth # Map right data to left boundary
      v[x, y] = v[nx(s) - 2 * depth + x, y]
    end
    for x in 1:depth # Map left data to right boundary
      v[nx(s) - x + 1, y] = v[2 * depth - x + 1, y]
    end
  end

  return v
end

function tb_boundary_coord_verts_map!(v, s::HasCubicalComplex, depth::Int)
  for x in 1:nx(s)
    for y in 1:depth # Map top data to bottom boundary
      v[x, y] = v[x, ny(s) - 2 * depth + y]
    end
    for y in 1:depth # Map bottom data to top boundary
      v[x, ny(s) - y + 1] = v[x, 2 * depth - y + 1]
    end
  end

  return v
end
