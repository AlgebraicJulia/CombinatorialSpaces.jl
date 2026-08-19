# ─────────────────────────────────────────────────────────────────────────────
#  SmoothingCache — kernel-based dual 0-form (quad) smoothing
#
#  Replaces the sparse-matrix pair smoothing_dual0(s,-c)*smoothing_dual0(s,c).
#  One SmoothingCache (built with a positive c_smooth) encodes both passes:
#    forward  pass: diag = 1 - c,  neighbor sign = +1
#    backward pass: diag = 1 + c,  neighbor sign = -1
#
#  Weight derivation (mirrors UniformMatrixDEC.smoothing_dual0):
#    c      = c_smooth / 2
#    tot_w  = (has_left+has_right)*inv_dx + (has_down+has_up)*inv_dy  (per quad)
#    w_nbr  = (c / tot_w) * inv_direction          (positive; stored once)
#    diag_fwd = 1 - c,   diag_bwd = 1 + c          (scalars, not per-quad)
#
#  The diagonal is constant across ALL quads because the matrix row always
#  sums to 1 regardless of how many neighbors exist.  Only the per-neighbor
#  weights vary at boundary quads (fewer neighbours → larger individual weight).
#  Missing neighbours get index 1 and weight 0; q_smask gates their contribution.
# ─────────────────────────────────────────────────────────────────────────────

struct SmoothingCache{IT <: AbstractVector{Int32}, FT <: AbstractVector{<:AbstractFloat}, MT <: AbstractVector{Int8}}
    nquads_  :: Int
    # Neighbor quad indices (Int32, dummy value 1 for non-existent boundary nbrs)
    q_left   :: IT
    q_right  :: IT
    q_down   :: IT
    q_up     :: IT
    # Per-quad neighbor weights for the FORWARD pass (positive c_smooth).
    # Backward pass uses the same magnitudes with negated sign.
    w_left   :: FT
    w_right  :: FT
    w_down   :: FT
    w_up     :: FT
    # Diagonal weights: constant across all quads, differ between passes.
    diag_fwd :: Float64   # = 1 - c_smooth/2
    diag_bwd :: Float64   # = 1 + c_smooth/2
    # Bitmask: bit0=left exists, bit1=right exists, bit2=down exists, bit3=up exists
    q_smask  :: MT
end
  
Adapt.@adapt_structure SmoothingCache
  
function SmoothingCache(s::UniformCubicalComplex2D{FT}, c_smooth::Real) where FT <: AbstractFloat
    n      = nquads(s)
    c      = FT(c_smooth) / FT(2.0)
    inv_dx = inv(dx(s))
    inv_dy = inv(dy(s))
    nqx    = nxq(s)
    nqy    = nyq(s)
  
    q_left  = Vector{Int32}(undef, n)
    q_right = Vector{Int32}(undef, n)
    q_down  = Vector{Int32}(undef, n)
    q_up    = Vector{Int32}(undef, n)
    w_left  = Vector{FT}(undef, n)
    w_right = Vector{FT}(undef, n)
    w_down  = Vector{FT}(undef, n)
    w_up    = Vector{FT}(undef, n)
    q_smask = Vector{Int8}(undef, n)
  
    for q in quads(s)
      x, y = quad_to_coord(s, q)
  
      has_left  = x > 1
      has_right = x < nqx
      has_down  = y > 1
      has_up    = y < nqy
  
      tot_w = (Int(has_left) + Int(has_right)) * inv_dx +
              (Int(has_down) + Int(has_up))    * inv_dy
  
      if tot_w > 0
        scale   = c / tot_w
        w_left[q]  = has_left  ? scale * inv_dx : zero(FT)
        w_right[q] = has_right ? scale * inv_dx : zero(FT)
        w_down[q]  = has_down  ? scale * inv_dy : zero(FT)
        w_up[q]    = has_up    ? scale * inv_dy : zero(FT)
      else
        w_left[q] = w_right[q] = w_down[q] = w_up[q] = zero(FT)
      end
  
      q_left[q]  = Int32(has_left  ? coord_to_quad(s, x - 1, y) : 1)
      q_right[q] = Int32(has_right ? coord_to_quad(s, x + 1, y) : 1)
      q_down[q]  = Int32(has_down  ? coord_to_quad(s, x, y - 1) : 1)
      q_up[q]    = Int32(has_up    ? coord_to_quad(s, x, y + 1) : 1)
  
      q_smask[q] = Int8(has_left) |
                   (Int8(has_right) << Int8(1)) |
                   (Int8(has_down)  << Int8(2)) |
                   (Int8(has_up)    << Int8(3))
    end
  
    return SmoothingCache(n,
      q_left, q_right, q_down, q_up,
      w_left, w_right, w_down, w_up,
      FT(1.0) - c, FT(1.0) + c,
      q_smask)
end
  
  # ── Kernel ────────────────────────────────────────────────────────────────────
  # `diag`     — scalar diagonal weight (diag_fwd or diag_bwd)
  # `nbr_sign` — scalar +1.0 (forward pass) or -1.0 (backward pass)
  
@kernel function kernel_smooth_dual0_cached!(res,
                                               @Const(q_left),  @Const(q_right),
                                               @Const(q_down),  @Const(q_up),
                                               @Const(w_left),  @Const(w_right),
                                               @Const(w_down),  @Const(w_up),
                                               @Const(q_smask),
                                               @Const(f),
                                               diag, nbr_sign)
    q = @index(Global)
    @inbounds begin
      mask = q_smask[q]
      z    = zero(eltype(f))
      nbrs = ifelse(Bool(mask & Int8(1)),                     w_left[q]  * f[q_left[q]],  z) +
             ifelse(Bool((mask >> Int8(1)) & Int8(1)),        w_right[q] * f[q_right[q]], z) +
             ifelse(Bool((mask >> Int8(2)) & Int8(1)),        w_down[q]  * f[q_down[q]],  z) +
             ifelse(Bool((mask >> Int8(3)) & Int8(1)),        w_up[q]    * f[q_up[q]],    z)
      res[q] = diag * f[q] + nbr_sign * nbrs
    end
end
  
  # ── Interface ─────────────────────────────────────────────────────────────────
  
function _smooth_dual0_pass!(res::AbstractVector{FT}, cache::SmoothingCache,
                                f::AbstractVector{FT}, diag::FT, sign::FT) where FT <: AbstractFloat
    backend = get_backend(f)
    kernel  = kernel_smooth_dual0_cached!(backend)
    kernel(res,
      cache.q_left, cache.q_right, cache.q_down, cache.q_up,
      cache.w_left, cache.w_right, cache.w_down, cache.w_up,
      cache.q_smask, f, diag, sign;
      ndrange = cache.nquads_)
    return res
end
  
  """
      smooth_dual0_fused!(res, tmp, cache, f)
  
  Two-pass smoothing (`res .= M_bwd * (M_fwd * f)`) using a pre-allocated
  intermediate buffer `tmp`.  `cache` encodes both passes: the forward pass
  uses `+c_smooth` weights and the backward pass uses `-c_smooth` weights.
  """
function smooth_dual0_fused!(res::AbstractVector{FT}, tmp::AbstractVector{FT},
                                cache::SmoothingCache,
                                f::AbstractVector{FT}) where FT <: AbstractFloat
    _smooth_dual0_pass!(tmp, cache, f,   cache.diag_fwd,  FT(1.0))
    _smooth_dual0_pass!(res, cache, tmp, cache.diag_bwd, -FT(1.0))
    return res
end
  
  """
      smooth_dual0_fused!(res, cache, f)
  
  Allocating wrapper around `smooth_dual0_fused!`.  Prefer the pre-allocated
  three-argument form in hot paths.
  """
function smooth_dual0_fused!(res::AbstractVector{FT},
                                cache::SmoothingCache,
                                f::AbstractVector{FT}) where FT <: AbstractFloat
    tmp = similar(f)
    return smooth_dual0_fused!(res, tmp, cache, f)
end
  
function smooth_dual0_fused(cache::SmoothingCache, f::AbstractVector{FT}) where FT <: AbstractFloat
    res = similar(f)
    return smooth_dual0_fused!(res, cache, f)
end
  
struct SmoothingCache3D{IT <: AbstractVector{Int32},
    FT <: AbstractVector,
    MT <: AbstractVector{Int8}}
    nboids_ :: Int

    # Neighbor boid indices (dummy = 1 for missing boundary neighbors)
    b_west  :: IT;  b_east  :: IT
    b_south :: IT;  b_north :: IT
    b_down  :: IT;  b_up    :: IT

    # Per-boid neighbor weights
    w_west  :: FT;  w_east  :: FT
    w_south :: FT;  w_north :: FT
    w_down  :: FT;  w_up    :: FT

    # Scalar diagonal weights (same for all boids, differ between passes)
    diag_fwd :: Float64   # = 1 - c_smooth/2
    diag_bwd :: Float64   # = 1 + c_smooth/2

    # Bits 0..5 → west, east, south, north, down, up exist
    b_smask :: MT
end

Adapt.@adapt_structure SmoothingCache3D

function SmoothingCache3D(s::UniformCubicalComplex3D{FT}, c_smooth::Real) where FT <: AbstractFloat
    nb_    = nboids(s)
    c      = FT(c_smooth) / FT(2.0)
    inv_dx = inv(dx(s))
    inv_dy = inv(dy(s))
    inv_dz = inv(dz(s))
    nxb_   = nxb(s);  nyb_ = nyb(s);  nzb_ = nzb(s)

    b_west  = Vector{Int32}(undef, nb_);  b_east  = Vector{Int32}(undef, nb_)
    b_south = Vector{Int32}(undef, nb_);  b_north = Vector{Int32}(undef, nb_)
    b_down  = Vector{Int32}(undef, nb_);  b_up    = Vector{Int32}(undef, nb_)

    w_west  = Vector{FT}(undef, nb_);  w_east  = Vector{FT}(undef, nb_)
    w_south = Vector{FT}(undef, nb_);  w_north = Vector{FT}(undef, nb_)
    w_down  = Vector{FT}(undef, nb_);  w_up    = Vector{FT}(undef, nb_)

    b_smask = Vector{Int8}(undef, nb_)

    for b in 1:nb_
        x, y, z = boid_to_coord(s, b)

        has_west  = x > 1;     has_east  = x < nxb_
        has_south = y > 1;     has_north = y < nyb_
        has_down  = z > 1;     has_up    = z < nzb_

        tot_w = (Int(has_west)  + Int(has_east))  * inv_dx +
                (Int(has_south) + Int(has_north)) * inv_dy +
                (Int(has_down)  + Int(has_up))    * inv_dz

        if tot_w > 0
            scale = c / tot_w
            w_west[b]  = has_west  ? scale * inv_dx : zero(FT)
            w_east[b]  = has_east  ? scale * inv_dx : zero(FT)
            w_south[b] = has_south ? scale * inv_dy : zero(FT)
            w_north[b] = has_north ? scale * inv_dy : zero(FT)
            w_down[b]  = has_down  ? scale * inv_dz : zero(FT)
            w_up[b]    = has_up    ? scale * inv_dz : zero(FT)
        else
            w_west[b] = w_east[b] = w_south[b] =
            w_north[b] = w_down[b] = w_up[b] = zero(FT)
        end

        b_west[b]  = Int32(has_west  ? coord_to_boid(s, x-1, y, z) : 1)
        b_east[b]  = Int32(has_east  ? coord_to_boid(s, x+1, y, z) : 1)
        b_south[b] = Int32(has_south ? coord_to_boid(s, x, y-1, z) : 1)
        b_north[b] = Int32(has_north ? coord_to_boid(s, x, y+1, z) : 1)
        b_down[b]  = Int32(has_down  ? coord_to_boid(s, x, y, z-1) : 1)
        b_up[b]    = Int32(has_up    ? coord_to_boid(s, x, y, z+1) : 1)

        b_smask[b] = Int8(has_west)              |
                    (Int8(has_east)  << Int8(1)) |
                    (Int8(has_south) << Int8(2)) |
                    (Int8(has_north) << Int8(3)) |
                    (Int8(has_down)  << Int8(4)) |
                    (Int8(has_up)    << Int8(5))
    end

    return SmoothingCache3D(nb_,
        b_west, b_east, b_south, b_north, b_down, b_up,
        w_west, w_east, w_south, w_north, w_down, w_up,
        FT(1.0) - c, FT(1.0) + c,
        b_smask)
end

@kernel function kernel_smooth_dual0_3d_cached!(res,
        @Const(b_west),  @Const(b_east),
        @Const(b_south), @Const(b_north),
        @Const(b_down),  @Const(b_up),
        @Const(w_west),  @Const(w_east),
        @Const(w_south), @Const(w_north),
        @Const(w_down),  @Const(w_up),
        @Const(b_smask),
        @Const(f),
        diag, nbr_sign)
    b = @index(Global)
    @inbounds begin
        mask = b_smask[b]
        z    = zero(eltype(f))
        nbrs =
            ifelse(Bool( mask        & Int8(1)), w_west[b]  * f[b_west[b]],  z) +
            ifelse(Bool((mask >> 1)  & Int8(1)), w_east[b]  * f[b_east[b]],  z) +
            ifelse(Bool((mask >> 2)  & Int8(1)), w_south[b] * f[b_south[b]], z) +
            ifelse(Bool((mask >> 3)  & Int8(1)), w_north[b] * f[b_north[b]], z) +
            ifelse(Bool((mask >> 4)  & Int8(1)), w_down[b]  * f[b_down[b]],  z) +
            ifelse(Bool((mask >> 5)  & Int8(1)), w_up[b]    * f[b_up[b]],    z)
        res[b] = diag * f[b] + nbr_sign * nbrs
    end
end

function _smooth_dual0_pass_3d!(res, cache::SmoothingCache3D, f, diag, sign)
    backend = get_backend(f)
    kernel_smooth_dual0_3d_cached!(backend)(
        res,
        cache.b_west, cache.b_east, cache.b_south, cache.b_north,
        cache.b_down, cache.b_up,
        cache.w_west, cache.w_east, cache.w_south, cache.w_north,
        cache.w_down, cache.w_up,
        cache.b_smask, f, diag, sign; ndrange = cache.nboids_)
    return res
end

function smooth_dual0_fused!(res::AbstractVector{FT}, tmp::AbstractVector{FT},
        cache::SmoothingCache3D, f::AbstractVector{FT}) where {FT}
    _smooth_dual0_pass_3d!(tmp, cache, f,   cache.diag_fwd,  FT(1.0))
    _smooth_dual0_pass_3d!(res, cache, tmp, cache.diag_bwd, -FT(1.0))
    return res
end

function smooth_dual0_fused!(res::AbstractVector{FT},
        cache::SmoothingCache3D, f::AbstractVector{FT}) where {FT}
    tmp = similar(f)
    return smooth_dual0_fused!(res, tmp, cache, f)
end

function smooth_dual0_fused(cache::SmoothingCache3D, f::AbstractVector{FT}) where {FT}
    res = similar(f)
    return smooth_dual0_fused!(res, cache, f)
end
