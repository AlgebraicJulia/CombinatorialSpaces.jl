module UniformKernelDEC

using KernelAbstractions
using ..CombinatorialSpaces
using ..CubicalCode:

# Kernel for exterior derivative of 0-forms (primal 0-forms to primal 1-forms)
@kernel function kernel_exterior_derivative_zero!(res, s, f)
  idx = @index(Global)
  x, y, align = edge_to_coord(s, idx)

  if align == X_ALIGN
    @inbounds res[coord_to_edge(s, x, y, align)] = f[tgt_x(s, x, y)] - f[src_x(s, x, y)]
  elseif align == Y_ALIGN
    @inbounds res[coord_to_edge(s, x, y, align)] = f[tgt_y(s, x, y)] - f[src_y(s, x, y)]
  end
end

# Kernel for exterior derivative of 1-forms (primal 1-forms to primal 2-forms)
@kernel function kernel_exterior_derivative_one!(res, s, f, padding)
  idx = @index(Global)
  x, y = idx

  b_xe, t_xe, l_ye, r_ye = quad_edges(x, y)
  @inbounds res[idx] = f[b_xe] - f[t_xe] - f[l_ye] + f[r_ye]
end

# Main interface functions

function exterior_derivative!(res, ::Val{0}, s::UniformCubicalComplex2D, f)
  kernel = kernel_exterior_derivative_zero!(get_backend(res))

  kernel(res, s, f; ndrange = size(res))
end

function exterior_derivative!(res, ::Val{1}, s::UniformCubicalComplex2D, f)
  kernel = kernel_exterior_derivative_one!(get_backend(res))

  kernel(res, s, f; ndrange = size(res))
end
