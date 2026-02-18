abstract type AbstractWENO end

# TODO: Check that provided k beta_r an integer
struct UniformWENO{n, FT <: AbstractFloat} <: AbstractWENO end

polyorder(weno::UniformWENO{n}) where n = (n - 1) ÷ 2

abstract type AbstractStencil end

struct RectStencil{n, FT} <: AbstractStencil where {n, FT <: AbstractFloat}
  f::SVector{n, FT}

  function RectStencil{n, FT}(args...) where {n, FT <: AbstractFloat}
    (isinteger(n) && n >= 0) || error("Invalid stencil size")
    new(SVector{n, FT}(args...))
  end
end

function substencil(weno::UniformWENO{n, <:Any}, s::RectStencil, ::Val{j}) where {n, j}
  return @view s.f[j:j + polyorder(weno)]
end

# Order, stencil number
ISW(::UniformWENO{5, FT}, ::Val{1}) where FT = FT.((10, -31, 11, 25, -19,  4))
ISW(::UniformWENO{5, FT}, ::Val{2}) where FT = FT.((4,  -13, 5,  13, -13,  4))
ISW(::UniformWENO{5, FT}, ::Val{3}) where FT = FT.((4,  -19, 11, 25, -31, 10))

CW(::UniformWENO{5, FT}, ::Val{1}) where FT = FT.((1/3, -7/6, 11/6))
CW(::UniformWENO{5, FT}, ::Val{2}) where FT = FT.((-1/6, 5/6, 1/3))
CW(::UniformWENO{5, FT}, ::Val{3}) where FT = FT.((1/3, 5/6, -1/6))

d_r(::UniformWENO{5, FT}) where FT = FT.((1/10, 6/10, 3/10))

#TODO: Cite Oceananigans.jl
ISW(::UniformWENO{9, FT}, ::Val{1}) where FT = FT.((1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658))
ISW(::UniformWENO{9, FT}, ::Val{2}) where FT = FT.((0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908))
ISW(::UniformWENO{9, FT}, ::Val{3}) where FT = FT.((0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908))
ISW(::UniformWENO{9, FT}, ::Val{4}) where FT = FT.((0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658))
ISW(::UniformWENO{9, FT}, ::Val{5}) where FT = FT.((0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918))

CW(::UniformWENO{9, FT}, ::Val{1}) where FT = FT.((1/5, -21/20, 137/60, -163/60, 137/60))
CW(::UniformWENO{9, FT}, ::Val{2}) where FT = FT.((-1/20, 17/60, -43/60, 77/60, 1/5))
CW(::UniformWENO{9, FT}, ::Val{3}) where FT = FT.((1/30, -13/60, 47/60, 9/20, -1/20))
CW(::UniformWENO{9, FT}, ::Val{4}) where FT = FT.((-1/20, 9/20, 47/60, -13/60, 1/30))
CW(::UniformWENO{9, FT}, ::Val{5}) where FT = FT.((1/5, 77/60, -43/60, 17/60, -1/20))

d_r(::UniformWENO{9, FT}) where FT = FT.((1/126, 10/63, 10/21, 20/63, 5/126))

function beta_r(weno::UniformWENO{5, FT}, stencil::RectStencil{<:Any, FT}, ::Val{r}) where {r, FT}

  s = substencil(weno, stencil, Val(r))
  w = ISW(weno, Val(r))

  return s[1] * (w[1] * s[1] + w[2] * s[2] + w[3] * s[3]) +
         s[2] * (w[4] * s[2] + w[5] * s[3]) +
         s[3] * (w[6] * s[3])
end

function beta_r(weno::UniformWENO{9, FT}, stencil::RectStencil{<:Any, FT}, ::Val{r}) where {r, FT}

  s = substencil(weno, stencil, Val(r))
  w = ISW(weno, Val(r))

  return s[1] * (w[1] * s[1] + w[2] * s[2] + w[3] * s[3] + w[4] * s[4] + w[5] * s[5]) +
         s[2] * (w[6] * s[2] + w[7] * s[3] + w[8] * s[4] + w[9] * s[5]) +
         s[3] * (w[10] * s[3] + w[11] * s[4] + w[12] * s[5])
         s[4] * (w[13] * s[4] + w[14] * s[5])
         s[5] * (w[15] * s[5])
end


function flux(weno::UniformWENO, stencil::RectStencil{<:Any, FT}, ::Val{r}) where {r, FT}
  res = FT(0)

  f = substencil(weno, stencil, Val(r))
  Cs = CW(weno, Val(r))
  for (i, fi) in enumerate(f)
    res += Cs[i] * fi
  end
  return res
end

# TODO: Implement improved WENO-Z
function WENO(weno::UniformWENO{<:Any, FT}, stencil::RectStencil{<:Any, FT}; eps = 1e-6) where FT <: AbstractFloat
  res = FT(0)

  k = polyorder(weno)

  d_rs = d_r(weno)
  tmp_ω = FT(0)
  ω = FT(0)
  for i in 1:k
    tmp_ω = d_rs[i] / (eps + beta_r(weno, stencil, Val(i)))^2
    ω += tmp_ω
    res += tmp_ω * flux(weno, stencil, Val(i))
  end

  res /= ω

  return res
end
