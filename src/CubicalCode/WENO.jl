using KernelAbstractions

abstract type AbstractWENO end

struct UniformWENO{n, FT <: AbstractFloat} <: AbstractWENO
  n::Int
end

UniformWENO(n::Int, FT = Float64) = UniformWENO{n, FT}(n)
polyorder(weno::UniformWENO) = (weno.n - 1) ÷ 2

# -----------------------------------------------------------------------------
# Coefficients
# -----------------------------------------------------------------------------

const ISW3_1 = (1.0, -2.0, 1.0)
const ISW3_2 = (1.0, -2.0, 1.0)

const CW3_1 = (-1 / 2, 3 / 2)
const CW3_2 = (1 / 2, 1 / 2)

const DR3 = (1 / 3, 2 / 3)

const ISW5_1 = (10.0, -31.0, 11.0, 25.0, -19.0, 4.0)
const ISW5_2 = (4.0, -13.0, 5.0, 13.0, -13.0, 4.0)
const ISW5_3 = (4.0, -19.0, 11.0, 25.0, -31.0, 10.0)

const CW5_1 = (1 / 3, -7 / 6, 11 / 6)
const CW5_2 = (-1 / 6, 5 / 6, 1 / 3)
const CW5_3 = (1 / 3, 5 / 6, -1 / 6)

const DR5 = (1 / 10, 6 / 10, 3 / 10)

const ISW7_1 = (2.107, -9.402, 7.042, -1.854, 11.003, -17.246, 4.642, 7.043, -3.882, 0.547)
const ISW7_2 = (0.547, -2.522, 1.922, -0.494, 3.443, -5.966, 1.602, 2.843, -1.642, 0.267)
const ISW7_3 = (0.267, -1.642, 1.602, -0.494, 2.843, -5.966, 1.922, 3.443, -2.522, 0.547)
const ISW7_4 = (0.547, -3.882, 4.642, -1.854, 7.043, -17.246, 7.042, 11.003, -9.402, 2.107)

const CW7_1 = (-1 / 4, 13 / 12, -23 / 12, 25 / 12)
const CW7_2 = (1 / 12, -5 / 12, 13 / 12, 1 / 4)
const CW7_3 = (-1 / 12, 7 / 12, 7 / 12, -1 / 12)
const CW7_4 = (1 / 4, 13 / 12, -5 / 12, 1 / 12)

const DR7 = (1 / 35, 12 / 35, 18 / 35, 4 / 35)

const ISW9_1 = (1.07918, -6.49501, 7.58823, -4.11487, 0.86329, 10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863, 4.82963, -2.08501, 0.22658)
const ISW9_2 = (0.22658, -1.40251, 1.65153, -0.88297, 0.18079, 2.42723, -6.11976, 3.37018, -0.70237, 4.06293, -4.64976, 0.99213, 1.38563, -0.60871, 0.06908)
const ISW9_3 = (0.06908, -0.51001, 0.67923, -0.38947, 0.08209, 1.04963, -2.99076, 1.79098, -0.38947, 2.31153, -2.99076, 0.67923, 1.04963, -0.51001, 0.06908)
const ISW9_4 = (0.06908, -0.60871, 0.99213, -0.70237, 0.18079, 1.38563, -4.64976, 3.37018, -0.88297, 4.06293, -6.11976, 1.65153, 2.42723, -1.40251, 0.22658)
const ISW9_5 = (0.22658, -2.08501, 3.64863, -2.88007, 0.86329, 4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)

const CW9_1 = (1 / 5, -21 / 20, 137 / 60, -163 / 60, 137 / 60)
const CW9_2 = (-1 / 20, 17 / 60, -43 / 60, 77 / 60, 1 / 5)
const CW9_3 = (1 / 30, -13 / 60, 47 / 60, 9 / 20, -1 / 20)
const CW9_4 = (-1 / 20, 9 / 20, 47 / 60, -13 / 60, 1 / 30)
const CW9_5 = (1 / 5, 77 / 60, -43 / 60, 17 / 60, -1 / 20)

const DR9 = (1 / 126, 10 / 63, 10 / 21, 20 / 63, 5 / 126)

for FT in [Float32, Float64]
  @eval begin
      @inline ISW(::UniformWENO{3, $FT}, ::Val{1}) = $(FT.(ISW3_1))
      @inline ISW(::UniformWENO{3, $FT}, ::Val{2}) = $(FT.(ISW3_2))
      @inline CW(::UniformWENO{3, $FT}, ::Val{1}) = $(FT.(CW3_1))
      @inline CW(::UniformWENO{3, $FT}, ::Val{2}) = $(FT.(CW3_2))
      @inline d_r(::UniformWENO{3, $FT}) = $(FT.(DR3))

      @inline ISW(::UniformWENO{5, $FT}, ::Val{1}) = $(FT.(ISW5_1))
      @inline ISW(::UniformWENO{5, $FT}, ::Val{2}) = $(FT.(ISW5_2))
      @inline ISW(::UniformWENO{5, $FT}, ::Val{3}) = $(FT.(ISW5_3))
      @inline CW(::UniformWENO{5, $FT}, ::Val{1}) = $(FT.(CW5_1))
      @inline CW(::UniformWENO{5, $FT}, ::Val{2}) = $(FT.(CW5_2))
      @inline CW(::UniformWENO{5, $FT}, ::Val{3}) = $(FT.(CW5_3))
      @inline d_r(::UniformWENO{5, $FT}) = $(FT.(DR5))

      @inline ISW(::UniformWENO{7, $FT}, ::Val{1}) = $(FT.(ISW7_1))
      @inline ISW(::UniformWENO{7, $FT}, ::Val{2}) = $(FT.(ISW7_2))
      @inline ISW(::UniformWENO{7, $FT}, ::Val{3}) = $(FT.(ISW7_3))
      @inline ISW(::UniformWENO{7, $FT}, ::Val{4}) = $(FT.(ISW7_4))
      @inline CW(::UniformWENO{7, $FT}, ::Val{1}) = $(FT.(CW7_1))
      @inline CW(::UniformWENO{7, $FT}, ::Val{2}) = $(FT.(CW7_2))
      @inline CW(::UniformWENO{7, $FT}, ::Val{3}) = $(FT.(CW7_3))
      @inline CW(::UniformWENO{7, $FT}, ::Val{4}) = $(FT.(CW7_4))
      @inline d_r(::UniformWENO{7, $FT}) = $(FT.(DR7))

      @inline ISW(::UniformWENO{9, $FT}, ::Val{1}) = $(FT.(ISW9_1))
      @inline ISW(::UniformWENO{9, $FT}, ::Val{2}) = $(FT.(ISW9_2))
      @inline ISW(::UniformWENO{9, $FT}, ::Val{3}) = $(FT.(ISW9_3))
      @inline ISW(::UniformWENO{9, $FT}, ::Val{4}) = $(FT.(ISW9_4))
      @inline ISW(::UniformWENO{9, $FT}, ::Val{5}) = $(FT.(ISW9_5))
      @inline CW(::UniformWENO{9, $FT}, ::Val{1}) = $(FT.(CW9_1))
      @inline CW(::UniformWENO{9, $FT}, ::Val{2}) = $(FT.(CW9_2))
      @inline CW(::UniformWENO{9, $FT}, ::Val{3}) = $(FT.(CW9_3))
      @inline CW(::UniformWENO{9, $FT}, ::Val{4}) = $(FT.(CW9_4))
      @inline CW(::UniformWENO{9, $FT}, ::Val{5}) = $(FT.(CW9_5))
      @inline d_r(::UniformWENO{9, $FT}) = $(FT.(DR9))
  end
end


# -----------------------------------------------------------------------------
# Core arithmetic helpers
# -----------------------------------------------------------------------------

@inline function beta2(s1::FT, s2::FT, c::NTuple{3, FT}) where {FT <: AbstractFloat}
  return c[1] * s1 * s1 + c[2] * s1 * s2 + c[3] * s2 * s2
end

@inline function beta3(s1::FT, s2::FT, s3::FT, c::NTuple{6, FT}) where {FT <: AbstractFloat}
  return c[1] * s1 * s1 + c[2] * s1 * s2 + c[3] * s1 * s3 +
         c[4] * s2 * s2 + c[5] * s2 * s3 +
         c[6] * s3 * s3
end

@inline function beta4(s1::FT, s2::FT, s3::FT, s4::FT, c::NTuple{10, FT}) where {FT <: AbstractFloat}
  return c[1] * s1 * s1 + c[2] * s1 * s2 + c[3] * s1 * s3 + c[4] * s1 * s4 +
         c[5] * s2 * s2 + c[6] * s2 * s3 + c[7] * s2 * s4 +
         c[8] * s3 * s3 + c[9] * s3 * s4 +
         c[10] * s4 * s4
end

@inline function beta5(s1::FT, s2::FT, s3::FT, s4::FT, s5::FT, c::NTuple{15, FT}) where {FT <: AbstractFloat}
  return c[1] * s1 * s1 + c[2] * s1 * s2 + c[3] * s1 * s3 + c[4] * s1 * s4 + c[5] * s1 * s5 +
         c[6] * s2 * s2 + c[7] * s2 * s3 + c[8] * s2 * s4 + c[9] * s2 * s5 +
         c[10] * s3 * s3 + c[11] * s3 * s4 + c[12] * s3 * s5 +
         c[13] * s4 * s4 + c[14] * s4 * s5 +
         c[15] * s5 * s5
end

@inline function weno3_point(fm1::FT, f0::FT, fp1::FT, eps::FT) where {FT <: AbstractFloat}
  weno = UniformWENO(3, FT)

  C1 = CW(weno, Val(1))
  C2 = CW(weno, Val(2))

  dr = d_r(weno)

  q1 = C1[1] * fm1 + C1[2] * f0
  q2 = C2[1] * f0 + C2[2] * fp1

  b1 = beta2(fm1, f0, ISW(weno, Val(1)))
  b2 = beta2(f0, fp1, ISW(weno, Val(2)))

  a1 = dr[1] / ((eps + b1) * (eps + b1))
  a2 = dr[2] / ((eps + b2) * (eps + b2))

  asum = a1 + a2
  return (a1 * q1 + a2 * q2) / asum
end

@inline function weno5_point(fm2::FT, fm1::FT, f0::FT, fp1::FT, fp2::FT, eps::FT) where {FT <: AbstractFloat}
  weno = UniformWENO(5, FT)

  C1 = CW(weno, Val(1))
  C2 = CW(weno, Val(2))
  C3 = CW(weno, Val(3))

  dr = d_r(weno)

  q1 = C1[1] * fm2 + C1[2] * fm1 + C1[3] * f0
  q2 = C2[1] * fm1 + C2[2] * f0 + C2[3] * fp1
  q3 = C3[1] * f0 + C3[2] * fp1 + C3[3] * fp2

  b1 = beta3(fm2, fm1, f0, ISW(weno, Val(1)))
  b2 = beta3(fm1, f0, fp1, ISW(weno, Val(2)))
  b3 = beta3(f0, fp1, fp2, ISW(weno, Val(3)))

  a1 = dr[1] / ((eps + b1) * (eps + b1))
  a2 = dr[2] / ((eps + b2) * (eps + b2))
  a3 = dr[3] / ((eps + b3) * (eps + b3))

  asum = a1 + a2 + a3
  return (a1 * q1 + a2 * q2 + a3 * q3) / asum
end

@inline function weno7_point(fm3::FT, fm2::FT, fm1::FT, f0::FT, fp1::FT, fp2::FT, fp3::FT, eps::FT) where {FT <: AbstractFloat}
  weno = UniformWENO(7, FT)

  C1 = CW(weno, Val(1))
  C2 = CW(weno, Val(2))
  C3 = CW(weno, Val(3))
  C4 = CW(weno, Val(4))

  dr = d_r(weno)

  q1 = C1[1] * fm3 + C1[2] * fm2 + C1[3] * fm1 + C1[4] * f0
  q2 = C2[1] * fm2 + C2[2] * fm1 + C2[3] * f0 + C2[4] * fp1
  q3 = C3[1] * fm1 + C3[2] * f0 + C3[3] * fp1 + C3[4] * fp2
  q4 = C4[1] * f0 + C4[2] * fp1 + C4[3] * fp2 + C4[4] * fp3

  b1 = beta4(fm3, fm2, fm1, f0, ISW(weno, Val(1)))
  b2 = beta4(fm2, fm1, f0, fp1, ISW(weno, Val(2)))
  b3 = beta4(fm1, f0, fp1, fp2, ISW(weno, Val(3)))
  b4 = beta4(f0, fp1, fp2, fp3, ISW(weno, Val(4)))

  a1 = dr[1] / ((eps + b1) * (eps + b1))
  a2 = dr[2] / ((eps + b2) * (eps + b2))
  a3 = dr[3] / ((eps + b3) * (eps + b3))
  a4 = dr[4] / ((eps + b4) * (eps + b4))

  asum = a1 + a2 + a3 + a4
  return (a1 * q1 + a2 * q2 + a3 * q3 + a4 * q4) / asum
end

@inline function weno9_point(fm4::FT, fm3::FT, fm2::FT, fm1::FT, f0::FT, fp1::FT, fp2::FT, fp3::FT, fp4::FT, eps::FT) where {FT <: AbstractFloat}
  weno = UniformWENO(9, FT)

  C1 = CW(weno, Val(1))
  C2 = CW(weno, Val(2))
  C3 = CW(weno, Val(3))
  C4 = CW(weno, Val(4))
  C5 = CW(weno, Val(5))

  dr = d_r(weno)

  q1 = C1[1] * fm4 + C1[2] * fm3 + C1[3] * fm2 + C1[4] * fm1 + C1[5] * f0
  q2 = C2[1] * fm3 + C2[2] * fm2 + C2[3] * fm1 + C2[4] * f0 + C2[5] * fp1
  q3 = C3[1] * fm2 + C3[2] * fm1 + C3[3] * f0 + C3[4] * fp1 + C3[5] * fp2
  q4 = C4[1] * fm1 + C4[2] * f0 + C4[3] * fp1 + C4[4] * fp2 + C4[5] * fp3
  q5 = C5[1] * f0 + C5[2] * fp1 + C5[3] * fp2 + C5[4] * fp3 + C5[5] * fp4

  b1 = beta5(fm4, fm3, fm2, fm1, f0, ISW(weno, Val(1)))
  b2 = beta5(fm3, fm2, fm1, f0, fp1, ISW(weno, Val(2)))
  b3 = beta5(fm2, fm1, f0, fp1, fp2, ISW(weno, Val(3)))
  b4 = beta5(fm1, f0, fp1, fp2, fp3, ISW(weno, Val(4)))
  b5 = beta5(f0, fp1, fp2, fp3, fp4, ISW(weno, Val(5)))

  a1 = dr[1] / ((eps + b1) * (eps + b1))
  a2 = dr[2] / ((eps + b2) * (eps + b2))
  a3 = dr[3] / ((eps + b3) * (eps + b3))
  a4 = dr[4] / ((eps + b4) * (eps + b4))
  a5 = dr[5] / ((eps + b5) * (eps + b5))

  asum = a1 + a2 + a3 + a4 + a5
  return (a1 * q1 + a2 * q2 + a3 * q3 + a4 * q4 + a5 * q5) / asum
end

# -----------------------------------------------------------------------------
# KernelAbstractions kernels
# -----------------------------------------------------------------------------

@kernel function kernel_weno3!(res, fm1, f0, fp1, eps)
  i = @index(Global)
  @inbounds res[i] = weno3_point(fm1[i], f0[i], fp1[i], eps)
end

@kernel function kernel_weno5!(res, fm2, fm1, f0, fp1, fp2, eps)
  i = @index(Global)
  @inbounds res[i] = weno5_point(fm2[i], fm1[i], f0[i], fp1[i], fp2[i], eps)
end

@kernel function kernel_weno7!(res, fm3, fm2, fm1, f0, fp1, fp2, fp3, eps)
  i = @index(Global)
  @inbounds res[i] = weno7_point(fm3[i], fm2[i], fm1[i], f0[i], fp1[i], fp2[i], fp3[i], eps)
end

@kernel function kernel_weno9!(res, fm4, fm3, fm2, fm1, f0, fp1, fp2, fp3, fp4, eps)
  i = @index(Global)
  @inbounds res[i] = weno9_point(fm4[i], fm3[i], fm2[i], fm1[i], f0[i], fp1[i], fp2[i], fp3[i], fp4[i], eps)
end

# -----------------------------------------------------------------------------
# Public batched API (kernel path)
# -----------------------------------------------------------------------------

function WENO!(res::AbstractVector{FT}, ::UniformWENO{3}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(res)
  kernel_weno3!(backend)(res, fm1, f0, fp1, eps; ndrange = length(res))
  return res
end

function WENO!(res::AbstractVector{FT}, ::UniformWENO{5}, fm2::AbstractVector{FT}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}, fp2::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(res)
  kernel_weno5!(backend)(res, fm2, fm1, f0, fp1, fp2, eps; ndrange = length(res))
  return res
end

function WENO!(res::AbstractVector{FT}, ::UniformWENO{7}, fm3::AbstractVector{FT}, fm2::AbstractVector{FT}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}, fp2::AbstractVector{FT}, fp3::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(res)
  kernel_weno7!(backend)(res, fm3, fm2, fm1, f0, fp1, fp2, fp3, eps; ndrange = length(res))
  return res
end

function WENO!(res::AbstractVector{FT}, ::UniformWENO{9}, fm4::AbstractVector{FT}, fm3::AbstractVector{FT}, fm2::AbstractVector{FT}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}, fp2::AbstractVector{FT}, fp3::AbstractVector{FT}, fp4::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(res)
  kernel_weno9!(backend)(res, fm4, fm3, fm2, fm1, f0, fp1, fp2, fp3, fp4, eps; ndrange = length(res))
  return res
end

function WENO(weno::UniformWENO{3}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(f0)
  res = KernelAbstractions.zeros(backend, FT, length(f0))
  return WENO!(res, weno, fm1, f0, fp1; eps = eps)
end

function WENO(weno::UniformWENO{5}, fm2::AbstractVector{FT}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}, fp2::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(f0)
  res = KernelAbstractions.zeros(backend, FT, length(f0))
  return WENO!(res, weno, fm2, fm1, f0, fp1, fp2; eps = eps)
end

function WENO(weno::UniformWENO{7}, fm3::AbstractVector{FT}, fm2::AbstractVector{FT}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}, fp2::AbstractVector{FT}, fp3::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(f0)
  res = KernelAbstractions.zeros(backend, FT, length(f0))
  return WENO!(res, weno, fm3, fm2, fm1, f0, fp1, fp2, fp3; eps = eps)
end

function WENO(weno::UniformWENO{9}, fm4::AbstractVector{FT}, fm3::AbstractVector{FT}, fm2::AbstractVector{FT}, fm1::AbstractVector{FT}, f0::AbstractVector{FT}, fp1::AbstractVector{FT}, fp2::AbstractVector{FT}, fp3::AbstractVector{FT}, fp4::AbstractVector{FT}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat}
  backend = get_backend(f0)
  res = KernelAbstractions.zeros(backend, FT, length(f0))
  return WENO!(res, weno, fm4, fm3, fm2, fm1, f0, fp1, fp2, fp3, fp4; eps = eps)
end

# Tuple convenience overloads.
WENO!(res::AbstractVector{FT}, weno::UniformWENO{3}, s::NTuple{3, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO!(res, weno, s[1], s[2], s[3]; eps = eps)

WENO!(res::AbstractVector{FT}, weno::UniformWENO{5}, s::NTuple{5, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO!(res, weno, s[1], s[2], s[3], s[4], s[5]; eps = eps)

WENO!(res::AbstractVector{FT}, weno::UniformWENO{7}, s::NTuple{7, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO!(res, weno, s[1], s[2], s[3], s[4], s[5], s[6], s[7]; eps = eps)

WENO!(res::AbstractVector{FT}, weno::UniformWENO{9}, s::NTuple{9, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO!(res, weno, s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]; eps = eps)

WENO(weno::UniformWENO{3}, s::NTuple{3, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO(weno, s[1], s[2], s[3]; eps = eps)

WENO(weno::UniformWENO{5}, s::NTuple{5, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO(weno, s[1], s[2], s[3], s[4], s[5]; eps = eps)

WENO(weno::UniformWENO{7}, s::NTuple{7, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO(weno, s[1], s[2], s[3], s[4], s[5], s[6], s[7]; eps = eps)

WENO(weno::UniformWENO{9}, s::NTuple{9, AbstractVector{FT}}; eps::FT = FT(1e-6)) where {FT <: AbstractFloat} =
  WENO(weno, s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9]; eps = eps)
