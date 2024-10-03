module CombinatorialSpacesCUDAExt

using CombinatorialSpaces
using CombinatorialSpaces.DiscreteExteriorCalculus: DiscreteHodge
using GeometryBasics
using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays
using Krylov
Point2D = Point2{Float64}
Point3D = Point3{Float64}
import CombinatorialSpaces: dec_wedge_product, dec_c_wedge_product, dec_c_wedge_product!, dec_p_wedge_product, 
dec_boundary, dec_differential, dec_dual_derivative, dec_hodge_star, dec_inv_hodge_star

# Wedge Product
#--------------

""" dec_wedge_product(::Type{Tuple{j,k}}, sd::HasDeltaSet, ::Type{Val{:CUDA}})

Return a function that computes the wedge product between a primal j-form and a primal k-form via CUDA.

See also: [`dec_p_wedge_product`](@ref), [`dec_c_wedge_product`](@ref), [`dec_c_wedge_product!`](@ref).
"""
function dec_wedge_product(::Type{Tuple{j,k}}, sd::HasDeltaSet, ::Type{Val{:CUDA}}) where {j,k}
  error("Unsupported combination of degrees $j and $k. Ensure that their sum is not greater than the degree of the complex.")
end

function dec_wedge_product(::Type{Tuple{0,0}}, sd::HasDeltaSet, ::Type{Val{:CUDA}})
  (f, g) -> f .* g
end

function dec_wedge_product(::Type{Tuple{k,0}}, sd::HasDeltaSet, ::Type{Val{:CUDA}}) where {k}
  wedge_cache = dec_p_wedge_product(Tuple{0,k}, sd, Val{:CUDA})
  (α, g) -> dec_c_wedge_product(Tuple{0,k}, g, α, wedge_cache, Val{:CUDA})
end

function dec_wedge_product(::Type{Tuple{0,k}}, sd::HasDeltaSet, ::Type{Val{:CUDA}}) where {k}
  wedge_cache = dec_p_wedge_product(Tuple{0,k}, sd, Val{:CUDA})
  (f, β) -> dec_c_wedge_product(Tuple{0,k}, f, β, wedge_cache, Val{:CUDA})
end

function dec_wedge_product(::Type{Tuple{1,1}}, sd::HasDeltaSet2D, ::Type{Val{:CUDA}})
  wedge_cache = dec_p_wedge_product(Tuple{1,1}, sd, Val{:CUDA})
  (α, β) -> dec_c_wedge_product(Tuple{1,1}, α, β, wedge_cache, Val{:CUDA})
end

#Preallocate values to compute a wedge product via CUDA.
function dec_p_wedge_product(::Type{Tuple{m,n}}, sd, ::Type{Val{:CUDA}}) where {m,n}
  wedge_cache = dec_p_wedge_product(Tuple{m,n}, sd)
  (CuArray.(wedge_cache[1:end-1])..., wedge_cache[end]) 
end

# Compute with a preallocated wedge product via CUDA.
function dec_c_wedge_product(::Type{Tuple{m,n}}, α, β, wedge_cache, ::Type{Val{:CUDA}}) where {m,n}
  res = CUDA.zeros(eltype(α), last(last(wedge_cache)))
  dec_c_wedge_product!(Tuple{m,n}, res, α, β, wedge_cache, Val{:CUDA})
end

# Compute with a preallocated wedge product via CUDA in-place.
function dec_c_wedge_product!(::Type{Tuple{j,k}}, res, α, β, wedge_cache, ::Type{Val{:CUDA}}) where {j,k}
  # Manually dispatch, since CUDA.jl kernels cannot.
  kernel = if (j,k) == (0,1)
    dec_cu_ker_c_wedge_product_01!
  elseif (j,k) == (0,2)
    dec_cu_ker_c_wedge_product_02!
  elseif (j,k) == (1,1)
    dec_cu_ker_c_wedge_product_11!
  else
    error("Unsupported combination of degrees $j and $k. Ensure that their sum is not greater than the degree of the complex.")
  end

  n_threads = CUDA.max_block_size.x
  n_blocks = min(ceil(Int, length(res) / n_threads), CUDA.max_grid_size.x)
  @cuda threads=n_threads blocks=n_blocks kernel(res, α, β, wedge_cache)
  res
end

function dec_cu_ker_c_wedge_product_01!(res::CuDeviceArray{T}, f, α, wedge_cache) where T
  p = wedge_cache[1]
  index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   
  stride = gridDim().x * blockDim().x
  i = index
  @inbounds while i <= Int32(length(res))
    res[i] = T(0.5) * α[i] * (f[p[i, Int32(1)]] + f[p[i, Int32(2)]])
    i += stride
  end
  nothing
end

function dec_cu_ker_c_wedge_product_02!(res, f, α, wedge_cache)
  p, c = wedge_cache[1], wedge_cache[2]
  i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   
  stride = gridDim().x * blockDim().x

  @inbounds while i <= Int32(length(res))
    c1, c2, c3, c4, c5, c6 = c[Int32(1),i], c[Int32(2),i], c[Int32(3),i], c[Int32(4),i], c[Int32(5),i], c[Int32(6),i]
    p1, p2, p3, p4, p5, p6 = p[Int32(1),i], p[Int32(2),i], p[Int32(3),i], p[Int32(4),i], p[Int32(5),i], p[Int32(6),i]
    res[i] = α[i] * (c1*f[p1] + c2*f[p2] + c3*f[p3] + c4*f[p4] + c5*f[p5] + c6*f[p6])
    i += stride
  end
  nothing
end

function dec_cu_ker_c_wedge_product_11!(res, α, β, wedge_cache)
  e, c = wedge_cache[1], wedge_cache[2]
  i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x   
  stride = gridDim().x * blockDim().x

  @inbounds while i <= Int32(length(res))
    e0, e1, e2 = e[Int32(1), i], e[Int32(2), i], e[Int32(3), i]
    c1, c2, c3 = c[Int32(1), i], c[Int32(2), i], c[Int32(3), i]
    ae0, ae1, ae2 = α[e0], α[e1], α[e2]
    be0, be1, be2 = β[e0], β[e1], β[e2]

    res[i] =
      (c1 * (ae2 * be1 - ae1 * be2) +
       c2 * (ae2 * be0 - ae0 * be2) +
       c3 * (ae1 * be0 - ae0 * be1))
    i += stride
  end
  nothing
end

# Boundary and Co-boundary
#-------------------------

""" dec_boundary(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}})

Compute a boundary matrix as a sparse CUDA matrix.
"""
dec_boundary(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_boundary(n, sd))

dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_boundary(n, sd))
dec_boundary(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_boundary(n, sd))

""" dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D, ::Type{Val{:CUDA}})

Compute a dual derivative matrix as a sparse CUDA matrix.
"""
dec_dual_derivative(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_dual_derivative(n, sd))

dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))
dec_dual_derivative(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_dual_derivative(n, sd))


""" dec_differential(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}})

Compute an exterior derivative matrix as a sparse CUDA matrix.
"""
dec_differential(n::Int, sd::HasDeltaSet, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_differential(n, sd))

dec_differential(n::Int, sd::EmbeddedDeltaDualComplex1D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_differential(n, sd))
dec_differential(n::Int, sd::EmbeddedDeltaDualComplex2D{Bool, float_type, _p} where _p, ::Type{Val{:CUDA}}) where float_type =
  CuSparseMatrixCSC{float_type}(dec_differential(n, sd))

# Hodge Star
#-----------

""" dec_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}})

Compute a Hodge star as a diagonal or generic sparse CUDA matrix.
"""
dec_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) = 
  dec_hodge_star(Val{n}, sd, h, Val{:CUDA})

dec_hodge_star(::Type{Val{n}}, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) where n =
  CuArray(dec_hodge_star(Val{n}, sd, h))

dec_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, h::GeometricHodge, ::Type{Val{:CUDA}}) =
  CuSparseMatrixCSC(dec_hodge_star(Val{1}, sd, h))

""" dec_inv_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}})

Compute an inverse Hodge star matrix as a diagonal CUDA matrix.
"""
dec_inv_hodge_star(n::Int, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) =
  dec_inv_hodge_star(Val{n}, sd, h, Val{:CUDA})

dec_inv_hodge_star(::Type{Val{n}}, sd::HasDeltaSet, h::DiscreteHodge, ::Type{Val{:CUDA}}) where n =
  CuArray(dec_inv_hodge_star(Val{n}, sd, h))

""" function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Type{Val{:CUDA}})

Return a function that computes the inverse geometric Hodge star of a primal 1-form via a GMRES solver.
"""
function dec_inv_hodge_star(::Type{Val{1}}, sd::EmbeddedDeltaDualComplex2D, ::GeometricHodge, ::Type{Val{:CUDA}})
  hdg = -1 * dec_hodge_star(1, sd, GeometricHodge(), Val{:CUDA})
  x -> Krylov.gmres(hdg, x, atol = 1e-14)[1]
end

end

