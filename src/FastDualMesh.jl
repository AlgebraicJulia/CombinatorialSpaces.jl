module FastDualMesh

using Catlab
using CombinatorialSpaces
using ..SimplicialSets
using ..SimplicialSets: CayleyMengerDet, operator_nz, ∂_nz, d_nz, cayley_menger, negate
using ..DiscreteExteriorCalculus
using ..DiscreteExteriorCalculus: relative_sign
using ..ArrayUtils
using ACSets.DenseACSets: attrtype_type

using StaticArrays: MVector

import CombinatorialSpaces.DiscreteExteriorCalculus: subdivide_duals!

export FastMesh, subdivide_duals!

struct FastMesh end

function (::Type{S})(t::AbstractDeltaSet1D, gen::FastMesh) where S <: AbstractDeltaDualComplex1D
  s = S()
  copy_primal_1D!(s, t)
  make_dual_simplices_1d!(s, gen)
  return s
end

function (::Type{S})(t::AbstractDeltaSet2D, gen::FastMesh) where S <: AbstractDeltaDualComplex2D
  s = S()
  copy_primal_2D!(s, t)
  make_dual_simplices_2d!(s, gen)
  return s
end

copy_primal_1D!(t::EmbeddedDeltaDualComplex1D{_o, _l, point_type} where {_o, _l}, s) where point_type = copy_primal_1D!(t, s, point_type)
copy_primal_1D!(t::EmbeddedDeltaDualComplex2D{_o, _l, point_type} where {_o, _l}, s) where point_type = copy_primal_1D!(t, s, point_type)

function copy_primal_1D!(t::HasDeltaSet1D, s::HasDeltaSet1D, ::Type{point_type}) where point_type
  v_range = add_parts!(t, :V, nv(s))
  e_range = add_parts!(t, :E, ne(s))

  src_point::Vector{point_type} = s[:point]
  tgt_point = @view t[:point]

  @inbounds for (i, v) in enumerate(v_range)
    tgt_point[v] = src_point[i]
  end

  src_v0 = @view s[:∂v0]
  src_v1 = @view s[:∂v1]

  tgt_v0 = @view t[:∂v0]
  tgt_v1 = @view t[:∂v1]

  @inbounds for (i, e) in enumerate(e_range)
    tgt_v0[e] = src_v0[i]
    tgt_v1[e] = src_v1[i]
  end

  if has_subpart(s, :edge_orientation)
    src_edge_orient = @view s[:edge_orientation]
    tgt_edge_orient = @view t[:edge_orientation]

    @inbounds for (i, e) in enumerate(e_range)
      tgt_edge_orient[e] = src_edge_orient[i]
    end
  end
end

function copy_primal_2D!(t::HasDeltaSet2D, s::HasDeltaSet2D)

  copy_primal_1D!(t, s)
  tri_range = add_parts!(t, :Tri, ntriangles(s))

  src_e0 = @view s[:∂e0]
  src_e1 = @view s[:∂e1]
  src_e2 = @view s[:∂e2]

  tgt_e0 = @view t[:∂e0]
  tgt_e1 = @view t[:∂e1]
  tgt_e2 = @view t[:∂e2]

  @inbounds for (i, tri) in enumerate(tri_range)
    tgt_e0[tri] = src_e0[i]
    tgt_e1[tri] = src_e1[i]
    tgt_e2[tri] = src_e2[i]
  end

  if has_subpart(s, :tri_orientation)
    src_tri_orient = @view s[:tri_orientation]
    tgt_tri_orient = @view t[:tri_orientation]

    @inbounds for (i, tri) in enumerate(tri_range)
      tgt_tri_orient[tri] = src_tri_orient[i]
    end
  end

end

make_dual_simplices_1d!(s::AbstractDeltaDualComplex1D, gen::FastMesh) = make_dual_simplices_1d!(s, E, gen)

function make_dual_simplices_1d!(sd::HasDeltaSet1D, ::Type{Simplex{n}}, gen::FastMesh) where n
  # Make dual vertices and edges.
  vc_range = add_parts!(sd, :DualV, nv(sd))
  ec_range = add_parts!(sd, :DualV, ne(sd))

  vcenter = @view sd[:vertex_center]
  ecenter = @view sd[:edge_center]

  vcenter .= vc_range
  ecenter .= ec_range

  v0 = @view sd[:∂v0]
  v1 = @view sd[:∂v1]

  D_edges_1 = add_parts!(sd, :DualE, ne(sd))
  D_edges_2 = add_parts!(sd, :DualE, ne(sd))

  d_v0_1 = @view sd[D_edges_1, :D_∂v0]
  d_v0_2 = @view sd[D_edges_2, :D_∂v0]

  d_v0_1 .= ec_range
  d_v0_2 .= ec_range

  d_v1 = @view sd[:D_∂v1]

  @inbounds for i in edges(sd)
    d_v1[D_edges_1[i]] = v0[i]
    d_v1[D_edges_2[i]] = v1[i]
  end
  
  # Orient elementary dual edges.
  if has_subpart(sd, :edge_orientation)
    # If orientations are not set, then set them here.

    if any(isnothing, sd[:edge_orientation])
      # 1-simplices only need to be orientable if the delta set is 1D.
      # (The 1-simplices in a 2D delta set need not represent a valid 1-Manifold.)
      if n == 1
        orient!(sd, E) || error("The 1-simplices of the given 1D delta set are non-orientable.")
      else
        sd[findall(isnothing, sd[:edge_orientation]), :edge_orientation] = zero(attrtype_type(sd, :Orientation))
      end
    end

    # TODO: Need to use for-loop here since orientations are by default "nothing"
    # Need to find a way around this
    edge_orient = @view sd[:edge_orientation]
    d_edge_orient = @view sd[:D_edge_orientation]
    @inbounds for i in edges(sd)
      d_edge_orient[D_edges_1[i]] = negate(edge_orient[i])
      d_edge_orient[D_edges_2[i]] = edge_orient[i]
    end
  end

  (D_edges_1, D_edges_2)
end

make_dual_simplices_1d!(sd::AbstractDeltaDualComplex2D, gen::FastMesh) = make_dual_simplices_1d!(sd, Tri, gen)

make_dual_simplices_2d!(sd::AbstractDeltaDualComplex2D, gen::FastMesh) = make_dual_simplices_2d!(sd, Tri, gen)

function make_dual_simplices_2d!(sd::HasDeltaSet2D, ::Type{Simplex{n}}, gen::FastMesh) where n
  # Make dual vertices and edges.
  D_edges01 = make_dual_simplices_1d!(sd, gen)

  tric_range = add_parts!(sd, :DualV, ntriangles(sd))
  tri_centers = @view sd[:tri_center]
  tri_centers .= tric_range

  e0 = @view sd[:∂e0]
  e1 = @view sd[:∂e1]
  e2 = @view sd[:∂e2]

  edge_centers = @view sd[:edge_center]

  D_edges12 = map((0,1,2)) do e
    add_parts!(sd, :DualE, ntriangles(sd))
  end

  D_edges02 = map((0,1,2)) do vs
    add_parts!(sd, :DualE, ntriangles(sd))
  end

  d_v0 = @view sd[:D_∂v0]
  d_v1 = @view sd[:D_∂v1]

  @inbounds for (i, de_12_0, de_12_1, de_12_2) in zip(triangles(sd), D_edges12...)
    d_v0[de_12_0] = tri_centers[i]
    d_v0[de_12_1] = tri_centers[i]
    d_v0[de_12_2] = tri_centers[i]

    d_v1[de_12_0] = edge_centers[e0[i]]
    d_v1[de_12_1] = edge_centers[e1[i]]
    d_v1[de_12_2] = edge_centers[e2[i]]
  end

  v0 = @view sd[:∂v0]
  v1 = @view sd[:∂v1]

  @inbounds for (i, de_02_0, de_02_1, de_02_2) in zip(triangles(sd), D_edges02...)
    d_v0[de_02_0] = tri_centers[i]
    d_v0[de_02_1] = tri_centers[i]
    d_v0[de_02_2] = tri_centers[i]

    d_v1[de_02_0] = v1[e1[i]]
    d_v1[de_02_1] = v0[e2[i]]
    d_v1[de_02_2] = v0[e1[i]]
  end

  # Make dual triangles.
  # Counterclockwise order in drawing with vertices 0, 1, 2 from left to right.
  D_triangle_schemas = ((0,1,1),(0,2,1),(1,2,0),(1,0,1),(2,0,0),(2,1,0))
  D_triangles = map((0,1,2,3,4,5)) do e
    add_parts!(sd, :DualTri, ntriangles(sd))
  end

  d_e0 = @view sd[:D_∂e0]
  d_e1 = @view sd[:D_∂e1]
  d_e2 = @view sd[:D_∂e2]

  triangle_edges = (e0, e1, e2)

  @inbounds for (i, schema) in enumerate(D_triangle_schemas)
    v,e,ev = schema
    d_e0[D_triangles[i]] .= D_edges12[e+1]
    d_e1[D_triangles[i]] .= D_edges02[v+1]
    d_e2[D_triangles[i]] .= view(D_edges01[ev+1], triangle_edges[e+1])
  end

  if has_subpart(sd, :tri_orientation)
    # If orientations are not set, then set them here.
    if any(isnothing, sd[:tri_orientation])
      # 2-simplices only need to be orientable if the delta set is 2D.
      # (The 2-simplices in a 3D delta set need not represent a valid 2-Manifold.)
      if n == 2
        orient!(sd, Tri) || error("The 2-simplices of the given 2D delta set are non-orientable.")
      else
        sd[findall(isnothing, sd[:tri_orientation]), :tri_orientation] = zero(attrtype_type(sd, :Orientation))
      end
    end

    # Orient elementary dual triangles.
    tri_orient = @view sd[:tri_orientation]
    D_tri_orient = @view sd[:D_tri_orientation]

    @inbounds for (i, D_tris) in enumerate(D_triangles)
      for (j, D_tri) in enumerate(D_tris)
        D_tri_orient[D_tri] = isodd(i) ? negate(tri_orient[j]) : tri_orient[j]
      end
    end

    # Orient elementary dual edges.
    D_edge_orient = @view sd[:D_edge_orientation]
    @inbounds for (e, D_edges) in enumerate(D_edges12)
      tri_edge_view = @view sd[triangle_edges[e], :edge_orientation]
      for (j, D_edge) in enumerate(D_edges)
        D_edge_orient[D_edge] = relative_sign(
          tri_edge_view[j], isodd(e-1) ? negate(tri_orient[j]) : tri_orient[j])
      end
    end
    # Remaining dual edges are oriented arbitrarily.
    lazy_edge_orient = @view sd[lazy(vcat, D_edges02...), :D_edge_orientation]
    lazy_edge_orient .= one(Bool)
  end

  D_triangles
end

function subdivide_duals!(sd::EmbeddedDeltaDualComplex1D, gen::FastMesh, args...)
  subdivide_duals_1d!(sd, gen, args...)
  precompute_volumes_1d!(sd, gen)
end

function subdivide_duals!(sd::EmbeddedDeltaDualComplex2D, gen::FastMesh, args...)
  subdivide_duals_2d!(sd, gen, args...)
  precompute_volumes_2d!(sd, gen)
end

subdivide_duals_1d!(sd::EmbeddedDeltaDualComplex1D{_o, _l, point_type} where {_o, _l}, fm::FastMesh, alg) where point_type = subdivide_duals_1d!(sd, fm, point_type, alg)
subdivide_duals_1d!(sd::EmbeddedDeltaDualComplex2D{_o, _l, point_type} where {_o, _l}, fm::FastMesh, alg) where point_type = subdivide_duals_1d!(sd, fm, point_type, alg)
subdivide_duals_2d!(sd::EmbeddedDeltaDualComplex2D{_o, _l, point_type} where {_o, _l}, fm::FastMesh, alg) where point_type = subdivide_duals_2d!(sd, fm, point_type, alg)

precompute_volumes_1d!(sd::EmbeddedDeltaDualComplex1D{_o, _l, point_type} where {_o, _l}, fm::FastMesh) where point_type = precompute_volumes_1d!(sd, fm, point_type)
precompute_volumes_1d!(sd::EmbeddedDeltaDualComplex2D{_o, _l, point_type} where {_o, _l}, fm::FastMesh) where point_type = precompute_volumes_1d!(sd, fm, point_type)
precompute_volumes_2d!(sd::EmbeddedDeltaDualComplex2D{_o, _l, point_type} where {_o, _l}, fm::FastMesh) where point_type = precompute_volumes_2d!(sd, fm, point_type)

function subdivide_duals_1d!(sd::HasDeltaSet1D, ::FastMesh, ::Type{point_type}, alg) where point_type

  v1 = @view sd[:∂v0]
  v2 = @view sd[:∂v1]

  e_centers = @view sd[:edge_center]
  dual_point_set = @view sd[:dual_point]

  points::Vector{point_type} = sd[:point]
  point_arr = MVector{2, point_type}(undef)

  @inbounds for v in vertices(sd)
    dual_point_set[v] = points[v]
  end
  @inbounds for e in edges(sd)
    point_arr[1] = points[v1[e]]
    point_arr[2] = points[v2[e]]
    dual_point_set[e_centers[e]] = geometric_center(point_arr, alg)
  end
end

function precompute_volumes_1d!(sd::HasDeltaSet1D, ::FastMesh, ::Type{point_type}) where point_type

  v0 = @view sd[:∂v0]
  v1 = @view sd[:∂v1]

  d_v0 = @view sd[:D_∂v0]
  d_v1 = @view sd[:D_∂v1]

  length_set = @view sd[:length]
  dual_length_set = @view sd[:dual_length]

  dual_points::Vector{point_type} = sd[:dual_point]
  point_arr = MVector{2, point_type}(undef)

  @inbounds for e in edges(sd)
    point_arr[1] = dual_points[v0[e]]
    point_arr[2] = dual_points[v1[e]]
    length_set[e] = volume(point_arr)
  end
  @inbounds for e in parts(sd, :DualE)
    point_arr[1] = dual_points[d_v0[e]]
    point_arr[2] = dual_points[d_v1[e]]
    dual_length_set[e] = volume(point_arr)
  end
end

function subdivide_duals_2d!(sd::HasDeltaSet2D, gen::FastMesh, ::Type{point_type}, alg) where point_type
  subdivide_duals_1d!(sd, gen, alg)

  e1 = @view sd[:∂e1]
  e2 = @view sd[:∂e2]

  v0 = @view sd[:∂v0]
  v1 = @view sd[:∂v1]

  tri_centers = @view sd[:tri_center]
  dual_point_set = @view sd[:dual_point]

  points::Vector{point_type} = sd[:point]
  point_arr = MVector{3, point_type}(undef)

  @inbounds for t in triangles(sd)
    point_arr[1] = points[v1[e1[t]]]
    point_arr[2] = points[v0[e2[t]]]
    point_arr[3] = points[v0[e1[t]]]

    dual_point_set[tri_centers[t]] = geometric_center(point_arr, alg)
  end
end

function precompute_volumes_2d!(sd::HasDeltaSet2D, gen::FastMesh, p::Type{point_type}) where point_type
  precompute_volumes_1d!(sd, gen)
  set_volumes!(Val{2}, sd, p)
  set_dual_volumes!(Val{2}, sd, p)
end

function set_volumes!(::Type{Val{2}}, sd::HasDeltaSet2D, ::Type{point_type}) where point_type

  area_set = @view sd[:area]

  e1 = @view sd[:∂e1]
  e2 = @view sd[:∂e2]

  v0 = @view sd[:∂v0]
  v1 = @view sd[:∂v1]

  points::Vector{point_type} = sd[:point]
  point_arr = MVector{3, point_type}(undef)

  @inbounds for t in triangles(sd)
    point_arr[1] = points[v1[e1[t]]]
    point_arr[2] = points[v0[e2[t]]]
    point_arr[3] = points[v0[e1[t]]]

    area_set[t] = volume(point_arr)
  end
end

function set_dual_volumes!(::Type{Val{2}}, sd::HasDeltaSet2D, ::Type{point_type}) where point_type

  dual_area_set = @view sd[:dual_area]

  d_e0 = @view sd[:D_∂e0]
  d_e1 = @view sd[:D_∂e1]
  d_e2 = @view sd[:D_∂e2]

  d_v0 = @view sd[:D_∂v0]
  d_v1 = @view sd[:D_∂v1]

  dual_points::Vector{point_type} = sd[:dual_point]
  point_arr = MVector{3, point_type}(undef)

  @inbounds for t in parts(sd, :DualTri)
    point_arr[1] = dual_points[d_v1[d_e1[t]]]
    point_arr[2] = dual_points[d_v0[d_e2[t]]]
    point_arr[3] = dual_points[d_v0[d_e0[t]]]

    dual_area_set[t] = volume(point_arr)
  end
end

end