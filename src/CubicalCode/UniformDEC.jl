module UniformDEC

# ── Core numerics ─────────────────────────────────────────────────────────────
using LinearAlgebra:      diag
using SparseArrays:       sparse, spdiagm
using KernelAbstractions: @kernel, @index, @Const, get_backend

import ..SimplicialSets: nv, ne, src, tgt, vertices, edges, interior, exterior_derivative
import ..DiscreteExteriorCalculus: point, dual_point, hodge_star, inv_hodge_star, codifferential, wedge_product, dual_derivative

# ── Mesh ──────────────────────────────────────────────────────────────────────
include("UniformMesh.jl")
include("UniformMesh3D.jl")

# ── DEC operators ─────────────────────────────────────────────────────────────
include("UniformMatrixDEC.jl")
include("UniformKernelDEC.jl")
include("UniformKernelDEC3D.jl")

export
  # ── Alignment types (UniformMesh.jl) ──────────────────────────────────────
  Align,
  X_ALIGN, Y_ALIGN, Z_ALIGN,
  GridSide, EASTWEST, NORTHSOUTH, UPDOWN, ALL,

  # ── Mesh types (UniformMesh.jl, UniformMesh3D.jl) ─────────────────────────
  AbstractCubicalComplex,
  AbstractCubicalComplex2D,
  AbstractEmbeddedCubicalComplex2D,
  AbstractCubicalComplex3D,
  AbstractEmbeddedCubicalComplex3D,
  UniformCubicalComplex2D,
  UniformCubicalComplex3D,
  PseudoCubicalMesh2D,
  PseudoCubicalMesh3D,
  UniformCubicalComplex,
  PseudoCubicalMesh,

  # ── Mesh dimension accessors ───────────────────────────────────────────────
  nx, ny, nz,
  nxr, nyr, nzr,
  dx, dy, dz,
  nv, nvr,
  ne,
  nxe, nye, nze,
  nxe_r, nye_r, nze_r,
  nxedges, nyedges, nzedges,
  nquads, nquadsr,
  nxq, nyq, nzq,
  nxqr, nyqr,
  nxyq, nxzq, nyzq,
  nxyquads, nxzquads, nyzquads,
  nboids, nboidsr,
  nxb, nyb, nzb,
  nxbr, nybr, nzbr,
  nxyb, nxzb, nyzb,

  # ── Halo accessors ─────────────────────────────────────────────────────────
  halo_west, halo_east,
  halo_south, halo_north,
  halo_down, halo_up,

  # ── Physical domain accessors ──────────────────────────────────────────────
  base_x, base_y, base_z,
  spacing,

  # ── Cell iterators ─────────────────────────────────────────────────────────
  vertices, edges, quads, boids,

  # ── Geometry ───────────────────────────────────────────────────────────────
  point, points,
  dual_point, dual_points,
  real_point, real_dual_point,

  # ── Index conversion ───────────────────────────────────────────────────────
  coord_to_vert,
  coord_to_edge,
  coord_to_quad,
  coord_to_boid,
  vert_to_coord,
  edge_to_coord,
  quad_to_coord,
  boid_to_coord,
  dual_edge_to_coord,
  real_coord_to_vert,
  real_coord_to_boid,
  real_coord_to_real_vert,
  coord_to_real_coord,
  vert_to_real_vert,

  # ── Incidence and orientation ──────────────────────────────────────────────
  src, tgt,
  quad_vertices, quad_edges,
  boid_vertices, boid_quads, boid_edges,
  edge_quads,
  quad_boids,
  vertex_edges,
  edge_boids,

  # ── Metric quantities ──────────────────────────────────────────────────────
  edge_len, dual_edge_len,
  quad_area, dual_quad_area,
  boid_volume, dual_boid_volume,
  dual_edge,
  dual_quad,

  # ── Edge/quad family views ─────────────────────────────────────────────────
  xedges, yedges, zedges,
  xyquads, xzquads, yzquads,

  # ── Boundary helpers ───────────────────────────────────────────────────────
  boundary_edges, boundary_edges_real,
  top_edges, bottom_edges, left_edges, right_edges,
  top_edges_real, bottom_edges_real,
  left_edges_real, right_edges_real,
  boundary_tangent_edges,
  down_tangent_edges, up_tangent_edges,
  south_tangent_edges, north_tangent_edges,
  west_tangent_edges, east_tangent_edges,
  boundary_quads,
  down_quads, up_quads,
  south_quads, north_quads,
  west_quads, east_quads,
  primal_boundary_vertices,
  primal_boundary_quads,
  primal_boundary_boids,
  ghost_quads,
  ghost_boids,

  # ── Predicates ─────────────────────────────────────────────────────────────
  is_edge_X_aligned, is_edge_Y_aligned, is_edge_Z_aligned,
  is_halo_vert, is_halo_quad,
  is_boundary_vert, is_boundary_edge,
  is_left_edge, is_right_edge,
  is_top_edge, is_bottom_edge,
  valid_xedge, valid_yedge, valid_zedge,
  valid_xyquad, valid_xzquad, valid_yzquad,
  valid_boid,

  # ── Interior extraction ────────────────────────────────────────────────────
  interior,

  # ── Offset helpers ─────────────────────────────────────────────────────────
  edge_vertex_offset,
  quad_edge_offset,
  quad_edge_offset_3D,
  boid_quad_offset,

  # ── Operators ───────────────
  exterior_derivative,
  exterior_derivative!,

  dual_derivative,
  dual_derivative!,

  d_beta,
  d_beta_mul,
  d_beta_mul!,

  no_flux_dual_derivative,
  no_flux_dual_derivative!,

  hodge_star,
  hodge_star!,

  inv_hodge_star,
  inv_hodge_star!,

  codifferential,
  codifferential!,

  dual_codifferential,
  dual_codifferential!,

  laplacian,
  laplacian!,

  dual_laplacian,
  dual_laplacian!,

  interpolate_dp,
  interpolate_dp!,

  wedge_product,
  wedge_product!,
  wedge_product_dd,
  wedge_product_dd!,
  wedge_product_pd,
  wedge_product_pd!,

  sharp_dd,
  sharp_dd!,

  flat_dp,
  flat_dp!,

  flat_dd,
  flat_dd!,

  UniformDECCache,
  UniformDECCache3D
end
