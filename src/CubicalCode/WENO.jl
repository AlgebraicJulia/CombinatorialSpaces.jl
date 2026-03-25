abstract type AbstractWENO end

# TODO: Check that provided k beta_r an integer
struct UniformWENO{n, FT <: AbstractFloat} <: AbstractWENO
  n::Int
end

UniformWENO(n::Int, FT = Float64) = UniformWENO{n, FT}(n)

polyorder(weno::UniformWENO) = (weno.n - 1) ÷ 2

abstract type AbstractStencil end

struct RectStencil{FT} <: AbstractStencil where FT <: AbstractFloat
  f::SVector{<:Any, FT}

  function RectStencil{FT}(n, args...) where FT <: AbstractFloat
    n >= 0 || error("Invalid stencil size")
    new(SVector(args...))
  end

  function RectStencil{FT}(vec::AbstractVector) where FT <: AbstractFloat
    n = length(vec)
    n >= 0 || error("Invalid stencil size")
    new(SVector(vec...))
  end
end

function substencil(weno::UniformWENO, s::RectStencil, j::Int)
  r = polyorder(weno)
  return s.f[j:j + r]
end

# TODO: Cite Oceananigans.jl

# WENO-5 (5th order, 3 substencils)
ISW(::UniformWENO{5, FT}, ::Val{1}) where FT = FT.((10, -31, 11, 25, -19,  4))
ISW(::UniformWENO{5, FT}, ::Val{2}) where FT = FT.((4,  -13, 5,  13, -13,  4))
ISW(::UniformWENO{5, FT}, ::Val{3}) where FT = FT.((4,  -19, 11, 25, -31, 10))

# WENO-5 reconstruction coefficients
CW(::UniformWENO{5, FT}, ::Val{1}) where FT = FT.((1/3, -7/6, 11/6))
CW(::UniformWENO{5, FT}, ::Val{2}) where FT = FT.((-1/6, 5/6, 1/3))
CW(::UniformWENO{5, FT}, ::Val{3}) where FT = FT.((1/3, 5/6, -1/6))

d_r(::UniformWENO{5, FT}) where FT = FT.((1/10, 6/10, 3/10))

# WENO-7 (7th order, 4 substencils)
ISW(::UniformWENO{7, FT}, ::Val{1}) where FT = FT.((2.107,  -9.402, 7.042, -1.854, 11.003,  -17.246,  4.642,  7.043,  -3.882, 0.547))
ISW(::UniformWENO{7, FT}, ::Val{2}) where FT = FT.((0.547,  -2.522, 1.922, -0.494,  3.443,  - 5.966,  1.602,  2.843,  -1.642, 0.267))
ISW(::UniformWENO{7, FT}, ::Val{3}) where FT = FT.((0.267,  -1.642, 1.602, -0.494,  2.843,  - 5.966,  1.922,  3.443,  -2.522, 0.547))
ISW(::UniformWENO{7, FT}, ::Val{4}) where FT = FT.((0.547,  -3.882, 4.642, -1.854,  7.043,  -17.246,  7.042, 11.003,  -9.402, 2.107))

# WENO-7 reconstruction coefficients
CW(::UniformWENO{7, FT}, ::Val{1}) where FT = FT.((-1/4, 13/12, -23/12, 25/12))
CW(::UniformWENO{7, FT}, ::Val{2}) where FT = FT.((1/12, -5/12, 13/12, 1/4))
CW(::UniformWENO{7, FT}, ::Val{3}) where FT = FT.((-1/12, 7/12, 7/12, -1/12))
CW(::UniformWENO{7, FT}, ::Val{4}) where FT = FT.((1/4, 13/12, -5/12, 1/12))

d_r(::UniformWENO{7, FT}) where FT = FT.((1/35, 12/35, 18/35, 4/35))

# WENO-9 (9th order, 5 substencils)
ISW(::UniformWENO{9, FT}, ::Val{1}) where FT = FT.((1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658))
ISW(::UniformWENO{9, FT}, ::Val{2}) where FT = FT.((0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908))
ISW(::UniformWENO{9, FT}, ::Val{3}) where FT = FT.((0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908))
ISW(::UniformWENO{9, FT}, ::Val{4}) where FT = FT.((0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658))
ISW(::UniformWENO{9, FT}, ::Val{5}) where FT = FT.((0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918))

# WENO-9 reconstruction coefficients
CW(::UniformWENO{9, FT}, ::Val{1}) where FT = FT.((1/5, -21/20, 137/60, -163/60, 137/60))
CW(::UniformWENO{9, FT}, ::Val{2}) where FT = FT.((-1/20, 17/60, -43/60, 77/60, 1/5))
CW(::UniformWENO{9, FT}, ::Val{3}) where FT = FT.((1/30, -13/60, 47/60, 9/20, -1/20))
CW(::UniformWENO{9, FT}, ::Val{4}) where FT = FT.((-1/20, 9/20, 47/60, -13/60, 1/30))
CW(::UniformWENO{9, FT}, ::Val{5}) where FT = FT.((1/5, 77/60, -43/60, 17/60, -1/20))

d_r(::UniformWENO{9, FT}) where FT = FT.((1/126, 10/63, 10/21, 20/63, 5/126))

function beta_r(weno::UniformWENO{n, FT}, stencil::RectStencil{FT}, i::Int) where {n, FT <: AbstractFloat}
    s = substencil(weno, stencil, i)

    w = ISW(weno, Val(i))
    k = length(s)

    beta = zero(FT)
    idx = 1

    @inbounds for i in 1:k
        for j in i:k
            beta += s[i] * w[idx] * s[j]
            idx += 1
        end
    end

    return beta
end

function flux(weno::UniformWENO, stencil::RectStencil{FT}, i::Int) where {FT <: AbstractFloat}
  res = zero(FT)

  f = substencil(weno, stencil, i)
  Cs = CW(weno, Val(i))
  @inbounds for (i, fi) in enumerate(f)
    res += Cs[i] * fi
  end
  return res
end

# TODO: Implement improved WENO-Z
function WENO(weno::UniformWENO{<:Any, FT}, stencil::RectStencil{FT}; eps::FT = FT(1e-6)) where FT <: AbstractFloat
  res = zero(FT)

  k = polyorder(weno)

  d_rs = d_r(weno)
  ω = zero(FT)
  @inbounds for i in 1:k
    tmp_ω = d_rs[i] / (eps + beta_r(weno, stencil, i))^2
    ω += tmp_ω
    res += tmp_ω * flux(weno, stencil, i)
  end

  res /= ω

  return res
end
