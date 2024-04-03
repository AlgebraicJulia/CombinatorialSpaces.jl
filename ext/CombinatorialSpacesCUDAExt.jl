module CombinatorialSpacesCUDAExt

using CombinatorialSpaces
using GeometryBasics
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays
Point2D = Point2{Float64}
Point3D = Point3{Float64}
import CombinatorialSpaces: dec_cu_c_wedge_product!, dec_cu_c_wedge_product, dec_cu_p_wedge_product, dec_cu_wedge_product,
dec_boundary, dec_differential, dec_dual_derivative, dec_hodge_star, dec_inv_hodge_star

function dec_cu_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet)
  (f, g) -> f .* g
end

function dec_cu_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet) where {k}
  val_pack = dec_cu_p_wedge_product(Tuple{0,k}, sd)
  (α, g) -> dec_cu_c_wedge_product(Tuple{0,k}, g, α, val_pack)
end

function dec_cu_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet) where {k}
  val_pack = dec_cu_p_wedge_product(Tuple{0,k}, sd)
  (f, β) -> dec_cu_c_wedge_product(Tuple{0,k}, f, β, val_pack)
end

function dec_cu_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D)
  val_pack = dec_cu_p_wedge_product(Tuple{1,1}, sd)
  (α, β) -> dec_cu_c_wedge_product(Tuple{1,1}, α, β, val_pack)
end

dec_cu_p_wedge_product(::Type{Tuple{m,n}}, sd) where {m,n} = begin
  val_pack = dec_p_wedge_product(Tuple{m,n}, sd)
  (CuArray.(val_pack[1:end-1])..., val_pack[end]) 
end

# TODO: Should add typing to the zeros call
dec_cu_c_wedge_product(::Type{Tuple{m,n}}, α, β, val_pack) where {m,n} = begin
  wedge_terms = CUDA.zeros(Float64, last(last(val_pack)))
  return dec_cu_c_wedge_product!(Tuple{m,n}, wedge_terms, α, β, val_pack)
end

function dec_cu_c_wedge_product!(::Type{Tuple{0,1}}, wedge_terms, f, α, val_pack)
  num_threads = CUDA.max_block_size.x
  num_blocks = min(ceil(Int, length(wedge_terms) / num_threads), CUDA.max_grid_size.x)

  @cuda threads=num_threads blocks=num_blocks dec_cu_ker_c_wedge_product_01!(wedge_terms, f, α, val_pack[1])
  return wedge_terms
end

function dec_cu_c_wedge_product!(::Type{Tuple{0,2}}, wedge_terms, f, α, val_pack)
  num_threads = CUDA.max_block_size.x
  num_blocks = min(ceil(Int, length(wedge_terms) / num_threads), CUDA.max_grid_size.x)

  @cuda threads=num_threads blocks=num_blocks dec_cu_ker_c_wedge_product_02!(wedge_terms, f, α, val_pack[1], val_pack[2])
  return wedge_terms
end

function dec_cu_c_wedge_product!(::Type{Tuple{1,1}}, wedge_terms, f, α, val_pack)
  num_threads = CUDA.max_block_size.x
  num_blocks = min(ceil(Int, length(wedge_terms) / num_threads), CUDA.max_grid_size.x)

  @cuda threads=num_threads blocks=num_blocks dec_cu_ker_c_wedge_product_11!(wedge_terms, f, α, val_pack[1], val_pack[2])
  return wedge_terms
end

function dec_cu_ker_c_wedge_product_01!(wedge_terms::CuDeviceArray{T}, f, α, primal_vertices) where T
  index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   
  stride = gridDim().x * blockDim().x
  i = index
  @inbounds while i <= Int32(length(wedge_terms))
    wedge_terms[i] = T(0.5) * α[i] * (f[primal_vertices[i, Int32(1)]] + f[primal_vertices[i, Int32(2)]])
    i += stride
  end
  return nothing
end

function dec_cu_ker_c_wedge_product_02!(wedge_terms::CuDeviceArray{T}, f, α, pv, coeffs) where T
  index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   
  stride = gridDim().x * blockDim().x
  i = index

  @inbounds while i <= Int32(length(wedge_terms))
    wedge_terms[i] = α[i] * (coeffs[Int32(1), i] * f[pv[Int32(1), i]] + coeffs[Int32(2), i] * f[pv[Int32(2), i]]
                                + coeffs[Int32(3), i] * f[pv[Int32(3), i]] + coeffs[Int32(4), i] * f[pv[Int32(4), i]]
                                + coeffs[Int32(5), i] * f[pv[Int32(5), i]] + coeffs[Int32(6), i] * f[pv[Int32(6), i]])
    i += stride
  end
  return nothing
end

function dec_cu_ker_c_wedge_product_11!(wedge_terms, α, β, e, coeffs)
  index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   
  stride = gridDim().x * blockDim().x
  i = index

  @inbounds while i <= Int32(length(wedge_terms))
      e0, e1, e2 = e[Int32(1), i], e[Int32(2), i], e[Int32(3), i]
      ae0, ae1, ae2 = α[e0], α[e1], α[e2]
      be0, be1, be2 = β[e0], β[e1], β[e2]

      c1, c2, c3 = coeffs[Int32(1), i], coeffs[Int32(2), i], coeffs[Int32(3), i]

      wedge_terms[i] = (c1 * (ae2 * be1 - ae1 * be2)
                      + c2 * (ae2 * be0 - ae0 * be2)
                      + c3 * (ae1 * be0 - ae0 * be1))
      i += stride
  end

  return nothing
end

dec_boundary(n::Int, sd::HasDeltaSet, ::Type{Val{CUDA}}) = CuSparseMatrixCSC(dec_boundary(n, sd))
dec_dual_derivative(n::Int, sd::HasDeltaSet, ::Type{Val{CUDA}}) = CuSparseMatrixCSC(dec_dual_derivative(n, sd))
dec_differential(n::Int, sd::HasDeltaSet, ::Type{Val{CUDA}}) = CuSparseMatrixCSC(dec_differential(n, sd))

dec_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge, ::Type{Val{CUDA}}) = CuArray(dec_hodge_star(n, sd, DiagonalHodge()))
dec_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge, ::Type{Val{CUDA}}) = CuArray(dec_hodge_star(n, sd, GeometricHodge()))
dec_hodge_star(::Type{Val{1}}, sd::HasDeltaSet, ::GeometricHodge, ::Type{Val{CUDA}}) = CuSparseMatrixCSC(dec_hodge_star(Val{1}, sd, GeometricHodge()))

dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge, ::Type{Val{CUDA}}) = CuArray(dec_inv_hodge_star(n, sd, DiagonalHodge()))
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge, ::Type{Val{CUDA}}) = CuArray(dec_inv_hodge_star(n, sd, GeometricHodge()))

#= function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Type{Val{CUDA}})
  hdg = -1 * dec_hodge_star(1, sd, GeometricHodge(), Val{CUDA})
  x -> Krylov.gmres(hdg, x)[1]
end =#

# TODO: Revisit this exterior derivative kernel code later
#= 
### Exterior Derivatives Here ###

function dec_p_derivbound(::Type{Val{0}}, sd::HasDeltaSet; transpose::Bool=false, negate::Bool=false)
  vec_size = 2 * ne(sd)

  # XXX: This is assuming that meshes don't have too many entries
  # TODO: This type should be settable by the user and default set to Int32
  I = Vector{Int32}(undef, vec_size)
  J = Vector{Int32}(undef, vec_size)

  V = Vector{Int8}(undef, vec_size)

  e_orient::Vector{Int8} = sd[:edge_orientation]
  for i in eachindex(e_orient)
      e_orient[i] = (e_orient[i] == 1 ? 1 : -1)
  end

  v0_list = @view sd[:∂v0]
  v1_list = @view sd[:∂v1]

  for i in edges(sd)
      j = 2 * i - 1

      I[j] = i
      I[j+1] = i

      J[j] = v0_list[i]
      J[j+1] = v1_list[i]

      sign_term = e_orient[i]

      V[j] = sign_term
      V[j+1] = -1 * sign_term
  end

  if (transpose)
      I, J = J, I
  end
  if (negate)
      V .= -1 .* V
  end

  (I, J, V)
end

function dec_differential_test!(::Type{Val{0}}, sd)
  _, J, V = dec_p_derivbound(Val{0}, sd)
  J_p = Array(reshape(J, 2, length(J) ÷ 2)')
  V_p = Array(reshape(V, 2, length(V) ÷ 2)')

  e_orient = @view sd[:edge_orientation]
  all_edge_orientations_same = all(e_orient .== e_orient[1])

  if(all_edge_orientations_same)
    if(e_orient[1] == -1)
      J_p .= J_p[:, [2,1]]
    end
    return (res, x) -> dec_c_differential_same!(res, x, J_p)
  else
    return (res, x) -> dec_c_differential_not_same!(res, x, J_p, V_p)
    return nothing
  end
end

function dec_c_differential_same!(res, f, indices)
  @inbounds for i in 1:length(res)
    ind1 = indices[i, 1]
    ind2 = indices[i, 2]
    res[i] = f[ind1] - f[ind2]
  end
end

function dec_c_differential_not_same!(res, f, indices, signs)
  @inbounds for i in 1:length(res)
    ind1 = indices[i, 1]
    ind2 = indices[i, 2]
    res[i] = signs[ind1] * (f[ind1] - f[ind2])
  end
end

function dec_differential_cu_test!(::Type{Val{0}}, sd)
  _, J, V = dec_p_derivbound(Val{0}, sd)
  J_p = CuArray(reshape(J, 2, length(J) ÷ 2)')
  V_p = CuArray(reshape(V, 2, length(V) ÷ 2)')

  e_orient = @view sd[:edge_orientation]
  all_edge_orientations_same = all(e_orient .== e_orient[1])

  if(all_edge_orientations_same)
    if(e_orient[1] == -1)
      J_p .= J_p[:, [2,1]]
    end
    return (res, x) -> dec_cu_c_differential_same!(res, x, J_p)
  else
    # return (res, x) -> dec_c_differential_not_same!(res, x, J_p, V_p)
  end
end

function dec_cu_c_differential_same!(res, x, J_p)
  num_threads = CUDA.max_block_size.x
  num_blocks = min(ceil(Int, length(res) / num_threads), CUDA.max_grid_size.x)

  @cuda threads=num_threads blocks=num_blocks dec_cu_ker_c_differential_same!(res, x, J_p)
end

function dec_cu_ker_c_differential_same!(res, f, indices)
  index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   
  stride = gridDim().x * blockDim().x
  i = index

  @inbounds while i <= Int32(length(res))
    res[i] = f[indices[i, 1]] - f[indices[i, 2]]
    
    i += stride
  end

  return nothing
end =#

end

