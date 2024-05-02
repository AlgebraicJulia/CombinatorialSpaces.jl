module CombinatorialSpacesCUDAExt

using CombinatorialSpaces
using GeometryBasics
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays
using Krylov
Point2D = Point2{Float64}
Point3D = Point3{Float64}
import CombinatorialSpaces: dec_wedge_product!, dec_c_wedge_product, dec_c_wedge_product!, dec_p_wedge_product, 
dec_boundary, dec_differential, dec_dual_derivative, dec_hodge_star, dec_inv_hodge_star

function dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet, ::Type{Val{:CUDA}})
  (f, g) -> f .* g
end

function dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet, ::Type{Val{:CUDA}}) where {k}
  val_pack = dec_p_wedge_product(Tuple{0,k}, sd, Val{:CUDA})
  (α, g) -> dec_c_wedge_product(Tuple{0,k}, g, α, val_pack, Val{:CUDA})
end

function dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet, ::Type{Val{:CUDA}}) where {k}
  val_pack = dec_p_wedge_product(Tuple{0,k}, sd, Val{:CUDA})
  (f, β) -> dec_c_wedge_product(Tuple{0,k}, f, β, val_pack, Val{:CUDA})
end

function dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D, ::Type{Val{:CUDA}})
  val_pack = dec_p_wedge_product(Tuple{1,1}, sd, Val{:CUDA})
  (α, β) -> dec_c_wedge_product(Tuple{1,1}, α, β, val_pack, Val{:CUDA})
end

dec_p_wedge_product(::Type{Tuple{m,n}}, sd, ::Type{Val{:CUDA}}) where {m,n} = begin
  val_pack = dec_p_wedge_product(Tuple{m,n}, sd)
  (CuArray.(val_pack[1:end-1])..., val_pack[end]) 
end

dec_c_wedge_product(::Type{Tuple{m,n}}, α, β, val_pack, ::Type{Val{:CUDA}}) where {m,n} = begin
  wedge_terms = CUDA.zeros(eltype(α), last(last(val_pack)))
  return dec_c_wedge_product!(Tuple{m,n}, wedge_terms, α, β, val_pack, Val{:CUDA})
end

function dec_c_wedge_product!(::Type{Tuple{0,1}}, wedge_terms, f, α, val_pack, ::Type{Val{:CUDA}})
  num_threads = CUDA.max_block_size.x
  num_blocks = min(ceil(Int, length(wedge_terms) / num_threads), CUDA.max_grid_size.x)

  @cuda threads=num_threads blocks=num_blocks dec_cu_ker_c_wedge_product_01!(wedge_terms, f, α, val_pack[1])
  return wedge_terms
end

function dec_c_wedge_product!(::Type{Tuple{0,2}}, wedge_terms, f, α, val_pack, ::Type{Val{:CUDA}})
  num_threads = CUDA.max_block_size.x
  num_blocks = min(ceil(Int, length(wedge_terms) / num_threads), CUDA.max_grid_size.x)

  @cuda threads=num_threads blocks=num_blocks dec_cu_ker_c_wedge_product_02!(wedge_terms, f, α, val_pack[1], val_pack[2])
  return wedge_terms
end

function dec_c_wedge_product!(::Type{Tuple{1,1}}, wedge_terms, f, α, val_pack, ::Type{Val{:CUDA}})
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

dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type = CuSparseMatrixCSC{float_type}(dec_boundary(n, sd))
dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type = CuSparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))
dec_differential(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type = CuSparseMatrixCSC{float_type}(dec_differential(n, sd))

dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type = CuSparseMatrixCSC{float_type}(dec_boundary(n, sd))
dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type = CuSparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))
dec_differential(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type = CuSparseMatrixCSC{float_type}(dec_differential(n, sd))

dec_boundary(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) = CuSparseMatrixCSC(dec_boundary(n, sd))
dec_dual_derivative(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) = CuSparseMatrixCSC(dec_dual_derivative(n, sd))
dec_differential(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) = CuSparseMatrixCSC(dec_differential(n, sd))

dec_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge, ::Type{Val{:CUDA}}) = CuArray(dec_hodge_star(n, sd, DiagonalHodge()))
dec_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge, ::Type{Val{:CUDA}}) = dec_hodge_star(Val{n}, sd, GeometricHodge(), Val{:CUDA})
dec_hodge_star(::Type{Val{n}}, sd::HasDeltaSet, ::GeometricHodge, ::Type{Val{:CUDA}}) where n = CuArray(dec_hodge_star(Val{n}, sd, GeometricHodge()))
dec_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Type{Val{:CUDA}}) = CuSparseMatrixCSC(dec_hodge_star(Val{1}, sd, GeometricHodge()))

dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::DiagonalHodge, ::Type{Val{:CUDA}}) = CuArray(dec_inv_hodge_star(n, sd, DiagonalHodge()))
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, ::GeometricHodge, ::Type{Val{:CUDA}}) = dec_inv_hodge_star(Val{n}, sd, GeometricHodge(), Val{:CUDA})
dec_inv_hodge_star(::Type{Val{n}}, sd::HasDeltaSet, ::GeometricHodge, ::Type{Val{:CUDA}}) where n = CuArray(dec_inv_hodge_star(Val{n}, sd, GeometricHodge()))

function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Type{Val{:CUDA}})
  hdg = -1 * dec_hodge_star(1, sd, GeometricHodge(), Val{:CUDA})
  x -> Krylov.gmres(hdg, x, atol = 1e-14)[1]
end

end

