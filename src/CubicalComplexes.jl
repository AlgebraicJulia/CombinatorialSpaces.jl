export CubicalComplex, EmbeddedCubicalComplex2D

using StaticArrays: @SVector, SVector
using GeometryBasics: Point3d
using LinearAlgebra: norm, cross

abstract type HasCubicalComplex end

# True orientation of edges will be from smaller to greater vertex index
# TODO: Add parametric types for Int indexing, real (length, area)
# TODO: Might need to add more lookups for neighborhood search (vertex -> edge, vertex -> face, edge -> face)
mutable struct EmbeddedCubicalComplex2D <: HasCubicalComplex
  nv::Int
  ne::Int
  nquads::Int

  point::AbstractVector

  ∂v0::AbstractVector{Int}
  ∂v1::AbstractVector{Int}
  edge_orientation::AbstractVector{Bool}
  length::AbstractVector{Real}

  edge_lookup::Dict{Tuple{Int, Int}, Int}

  ∂e0::AbstractVector{Int}
  ∂e1::AbstractVector{Int}
  ∂e2::AbstractVector{Int}
  ∂e3::AbstractVector{Int}
  quad_orientation::AbstractVector{Bool}
  area::AbstractVector{Real}

  function EmbeddedCubicalComplex2D()
    return new(0,0,0, # Total count
      Point3d[], # For vertices
      Int64[],Int64[],Bool[],Float64[], # For edges
      Dict{Tuple{Int64, Int64}, Int64}(), # For edge lookup
      Int64[],Int64[],Int64[],Int64[],Bool[],Float64[]) # For quads
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

function add_edge!(s::HasCubicalComplex, v0::Int, v1::Int, o::Bool = false)
  @assert has_vertex(s, v0) && has_vertex(s, v1)
  @assert length(unique(SVector(v0, v1))) == 2

  e_idx = inc_ne!(s)
  v0, v1 = sort(SVector(v0, v1))

  has_edge(s, v0, v1) && return find_edge(s, v0, v1)

  push!(s.∂v0, v0); push!(s.∂v1, v1); push!(s.edge_orientation, o)
  push!(s.length, norm(point(s, v1) - point(s, v0)))
  push!(s.edge_lookup, (v0, v1) => e_idx)
  return e_idx
end

# TODO: This should be using a dictionary
has_edge(s::HasCubicalComplex, v0::Int, v1::Int) = has_edge(s, Tuple(SVector(v0,v1)))
has_edge(s::HasCubicalComplex, vs::Tuple{Int, Int}) = haskey(s.edge_lookup, sort(vs))

find_edge(s::HasCubicalComplex, v0::Int, v1::Int) = find_edge(s, Tuple(SVector(v0,v1)))
find_edge(s::HasCubicalComplex, vs::Tuple{Int, Int}) = get(s.edge_lookup, sort(vs), 0)

edge_vertices(s::HasCubicalComplex, e::Int) = SVector(s.∂v0, s.∂v1)

# TODO: Edges might want to have a better defined order, based on vertex ordering
function glue_quad!(s::HasCubicalComplex, v0::Int, v1::Int, v2::Int, v3::Int, o::Bool = false)
  @assert has_vertex(s, v0) && has_vertex(s, v1) && has_vertex(s, v2) && has_vertex(s, v3)
  @assert length(unique(SVector(v0, v1, v2, v3))) == 4

  q_idx = inc_nquads!(s)
  es = Int[] # TODO: Make this better

  for pair in SVector((v0, v1), (v1, v2), (v2, v3), (v3, v0))
    e = add_edge!(s, pair...)
    e != ne(s)
    push!(es, e)
  end

  # TODO: Add quad lookup to prevent redundant quads
  push!(s.∂e0, es[1]); push!(s.∂e1, es[2]); push!(s.∂e2, es[2]); push!(s.∂e3, es[3]);
  push!(s.quad_orientation, o)
  push!(s.area, norm(cross(point(s, v1) - point(s, v0), point(s, v3) - point(s, v0))))

  return s
end

# TODO: Take first quad and orient neighbors accordingly
function orient!()
  return nothing
end

quad_edges(s::HasCubicalComplex, q::Int) = SVector(s.∂e0, s.∂e1, s.∂e2, s.∂e3)

edge_length(s, e::Int) = getindex(s.length, e)
quad_area(s, q::Int) = getindex(s.area, q)
