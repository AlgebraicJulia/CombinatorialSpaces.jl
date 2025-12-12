export CubicalComplex, EmbeddedCubicalComplex2D

using StaticArrays: @SVector, SVector
using GeometryBasics: Point2d, Point3d, QuadFace
using LinearAlgebra: norm, cross
using SparseArrays

import Base.show
import GeometryBasics.Mesh

using Makie
import Makie: convert_arguments

abstract type HasCubicalComplex end

# True orientation of edges will be from smaller to greater vertex index
# TODO: Add parametric types for Int indexing, real (length, area)
# TODO: Might need to add more lookups for neighborhood search (vertex -> edge, vertex -> face, edge -> face)
mutable struct EmbeddedCubicalComplex2D <: HasCubicalComplex
  nv::Int
  ne::Int
  nquads::Int

  point::AbstractVector

  ∂v0::AbstractVector{Int} # tgt
  ∂v1::AbstractVector{Int} # src
  length::AbstractVector{Real}

  ∂e0::AbstractVector{Int}
  ∂e1::AbstractVector{Int}
  ∂e2::AbstractVector{Int}
  ∂e3::AbstractVector{Int}
  area::AbstractVector{Real}

  vertex_edge_lookup::Dict{SVector{2, Int}, Int}
  quad_vertex_lookup::Dict{Int, SVector{4, Int}}
  quad_edge_orient_lookup::Dict{Int, SVector{4, Int}}

  function EmbeddedCubicalComplex2D()
    return new(0,0,0, # Total count
      Point3d[], # For vertices
      Int32[],Int32[],Float64[], # For edges
      Int32[],Int32[],Int32[],Int32[],Float64[], # For quads
      Dict{SVector{2, Int32}, Int32}(), # For edge lookup
      Dict{Int, SVector{4, Int32}}(),
      Dict{Int, SVector{4, Int32}}()) # For quad vertex ordering lookup
  end
end

function Base.show(io::IO, s::HasCubicalComplex)
  println(io, "Mesh Information:\n")
  println(io, "Vertices: $(nv(s))")
  println(io, "Edges   : $(ne(s))")
  println(io, "Quads   : $(nquads(s))\n")

  println(io, "Points:")
  for (i, p) in enumerate(points(s))
    println(io, "$i: $p")
  end

  println(io, "\nEdges:")
  for i in edges(s)
    println(io, "$i: $(edge_vertices(s, i))")
  end

  println(io, "\nQuads:")
  for i in quadrilaterals(s)
    println(io, "$i: $(quad_edges(s, i))")
  end
end

nv(s::HasCubicalComplex) = s.nv
ne(s::HasCubicalComplex) = s.ne
nquads(s::HasCubicalComplex) = s.nquads

vertices(s::HasCubicalComplex) = 1:nv(s)
edges(s::HasCubicalComplex) = 1:ne(s)
quadrilaterals(s::HasCubicalComplex) = 1:nquads(s)

inc_nv!(s::HasCubicalComplex) = return (s.nv += 1)
inc_ne!(s::HasCubicalComplex) = return (s.ne += 1)
inc_nquads!(s::HasCubicalComplex) = return (s.nquads += 1)

function add_vertex!(s::HasCubicalComplex, p)
  v = inc_nv!(s)
  push!(s.point, p)
  return v
end

has_vertex(s::HasCubicalComplex, v::Int) = (v <= nv(s))

function point(s::HasCubicalComplex, v::Int)
  @assert has_vertex(s, v)
  return getindex(s.point, v)
end

points(s::HasCubicalComplex) = s.point

function add_edge!(s::HasCubicalComplex, v0::Int, v1::Int)
  @assert has_vertex(s, v0) && has_vertex(s, v1)

  v_list = edge_from_pair(v0, v1)
  @assert length(unique(v_list)) == 2

  has_edge(s, v_list) && return find_edge(s, v_list)

  e_idx = inc_ne!(s)
  push!(s.∂v0, v0); push!(s.∂v1, v1);
  push!(s.length, norm(point(s, v1) - point(s, v0)))
  push!(s.vertex_edge_lookup, v_list => e_idx)
  return e_idx
end

edge_from_pair(v0::Int, v1::Int) = sort(SVector(v0, v1))

has_edge(s::HasCubicalComplex, v0::Int, v1::Int) = has_edge(s, edge_from_pair(v0,v1))
has_edge(s::HasCubicalComplex, vs::SVector{2, Int}) = haskey(s.vertex_edge_lookup, vs)

find_edge(s::HasCubicalComplex, v0::Int, v1::Int) = find_edge(s, edge_from_pair(v0,v1))
find_edge(s::HasCubicalComplex, vs::SVector{2, Int}) = get(s.vertex_edge_lookup, vs, 0)

tgt(s::HasCubicalComplex, e::Int) = SVector(getindex(s.∂v0, e))
src(s::HasCubicalComplex, e::Int) = SVector(getindex(s.∂v1, e))
edge_vertices(s::HasCubicalComplex, e::Int) = SVector(getindex(s.∂v0, e), getindex(s.∂v1, e))

edge_length(s, e::Int) = getindex(s.length, e)

# Order of vertices must be given in counterclockwise order
function glue_quad!(s::HasCubicalComplex, v0::Int, v1::Int, v2::Int, v3::Int, o::Bool = true)
  @assert has_vertex(s, v0) && has_vertex(s, v1) && has_vertex(s, v2) && has_vertex(s, v3)

  list = SVector(v0, v1, v2, v3)
  @assert length(unique(list)) == 4


  q_idx = inc_nquads!(s)
  push!(s.quad_vertex_lookup, q_idx => list)

  # TODO: Make this more performant
  list = SVector(v0, v1, v2, v3, v0)
  es = MVector{4, Int}(0,0,0,0)
  orients = MVector{4, Int}(0,0,0,0)
  for i in 1:4
    pair = edge_from_pair(list[i], list[i+1])

    es[i] = add_edge!(s, pair...)
    orients[i] = (pair == SVector(list[i], list[i+1])) ? 1 : -1
  end

  # TODO: Add quad lookup to prevent redundant quads
  push!(s.∂e0, es[1]); push!(s.∂e1, es[2]); push!(s.∂e2, es[3]); push!(s.∂e3, es[4]);
  push!(s.area, norm(cross(point(s, v1) - point(s, v0), point(s, v3) - point(s, v0))))

  push!(s.quad_edge_orient_lookup, q_idx => orients)

  return s
end

quad_edges(s::HasCubicalComplex, q::Int) = SVector(getindex(s.∂e0, q), getindex(s.∂e1, q), getindex(s.∂e2, q), getindex(s.∂e3, q))
quad_area(s::HasCubicalComplex, q::Int) = getindex(s.area, q)
quad_vertices(s::HasCubicalComplex, q::Int) = get(s.quad_vertex_lookup, q, SVector(0,0,0,0))
quad_edge_orients(s::HasCubicalComplex, q::Int) = get(s.quad_edge_orient_lookup, q, SVector(0,0,0,0))

### PLOTTING ###

function GeometryBasics.Mesh(s::HasCubicalComplex)
  ps = map(q -> point(s, q), vertices(s))
  qs = map(q -> QuadFace{Int}(quad_vertices(s, q)), quadrilaterals(s))
  GeometryBasics.Mesh(ps, qs)
end

function convert_arguments(P::Union{Type{<:Makie.Wireframe},
                                    Type{<:Makie.Mesh},
                                    Type{<:Makie.Scatter}},
                           s::HasCubicalComplex)
  convert_arguments(P, GeometryBasics.Mesh(s))
end

function convert_arguments(P::Type{<:Makie.LineSegments}, s::HasCubicalComplex)
  edge_positions = zeros(ne(s)*2,3)
  for e in edges(s)
    edge_positions[2*e-1,:] = point(s, src(s, e))
    edge_positions[2*e,:] = point(s, tgt(s, e))
  end
  convert_arguments(P, edge_positions)
end

plottype(::HasCubicalComplex) = GeometryBasics.Mesh

### DEFAULT MESHES ###

function construct_grid(lx::Real, ly::Real, nx::Int, ny::Int)
  s = EmbeddedCubicalComplex2D()
  for y in range(0, ly, length = ny)
    for x in range(0, lx, length = nx)
      add_vertex!(s, Point3d(x, y, 0))
    end
  end

  for j in 1:ny - 1
    for i in 1:nx - 1
      base_idx = (j - 1) * nx + i
      glue_quad!(s, base_idx, base_idx + 1, base_idx + nx + 1, base_idx + nx)
    end
  end

  return s
end

### DEC OPERATORS ###

function exterior_derivative(::Val{0}, s::HasCubicalComplex)

  tot = 2 * ne(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  for e in edges(s)
    idx = 2 * e - 1
    v0, v1 = edge_vertices(s, e)

    I[idx] = e
    I[idx + 1] = e

    J[idx] = v0
    J[idx + 1] = v1

    V[idx] = 1
    V[idx + 1] = -1
  end

  return sparse(I, J, V)
end

function exterior_derivative(::Val{1}, s::HasCubicalComplex)

  tot = 4 * nquads(s)
  I, J = zeros(Float64, tot), zeros(Float64, tot)
  V = zeros(Float64, tot)

  for q in quadrilaterals(s)
    idx = 4 * q - 3
    orients = get(s.quad_edge_orient_lookup, q, SVector(0,0,0,0))
    for (i, e) in enumerate(quad_edges(s, q))
      j = idx + i - 1
      I[j] = q
      J[j] = e
      V[j] = getindex(orients, i)
    end
  end

  return sparse(I, J, V)
end
