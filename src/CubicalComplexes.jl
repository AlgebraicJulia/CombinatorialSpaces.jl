export CubicalComplex, EmbeddedCubicalComplex2D

using StaticArrays: @SVector, SVector
using GeometryBasics: Point3d
using LinearAlgebra: norm, cross
using SparseArrays

import Base.show

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
  length::AbstractVector{Real}

  ∂e0::AbstractVector{Int}
  ∂e1::AbstractVector{Int}
  ∂e2::AbstractVector{Int}
  ∂e3::AbstractVector{Int}
  quad_orientation::AbstractVector{Bool}
  area::AbstractVector{Real}

  # TODO: Change to Tuple to SVector?
  vertex_edge_lookup::Dict{SVector{2, Int}, Int}
  quad_vertex_lookup::Dict{Int, SVector{4, Int}}
  function EmbeddedCubicalComplex2D()
    return new(0,0,0, # Total count
      Point3d[], # For vertices
      Int64[],Int64[],Float64[], # For edges
      Int64[],Int64[],Int64[],Int64[],Bool[],Float64[], # For quads
      Dict{SVector{2, Int64}, Int64}(), # For edge lookup
      Dict{Int, SVector{4, Int64}}()) # For quad vertex ordering lookup
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
    println(io, "$i: $(quad_vertices(s, i))")
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

# TODO: This should be using a dictionary
has_edge(s::HasCubicalComplex, v0::Int, v1::Int) = has_edge(s, SVector(v0,v1))
has_edge(s::HasCubicalComplex, vs::SVector{2, Int}) = haskey(s.vertex_edge_lookup, sort(vs))

find_edge(s::HasCubicalComplex, v0::Int, v1::Int) = find_edge(s, SVector(v0,v1))
find_edge(s::HasCubicalComplex, vs::SVector{2, Int}) = get(s.vertex_edge_lookup, sort(vs), 0)

edge_vertices(s::HasCubicalComplex, e::Int) = SVector(getindex(s.∂v0, e), getindex(s.∂v1, e))

edge_length(s, e::Int) = getindex(s.length, e)

# TODO: Edges might want to have a better defined order, based on vertex ordering
# Order of vertices must be given in counterclockwise order
function glue_quad!(s::HasCubicalComplex, v0::Int, v1::Int, v2::Int, v3::Int, o::Bool = true)
  @assert has_vertex(s, v0) && has_vertex(s, v1) && has_vertex(s, v2) && has_vertex(s, v3)

  v_list = SVector(v0, v1, v2, v3)
  @assert length(unique(v_list)) == 4

  q_idx = inc_nquads!(s)
  push!(s.quad_vertex_lookup, q_idx => v_list)

  es = Int[] # TODO: Make this better

  list = sort(SVector(edge_from_pair(v0, v1), edge_from_pair(v1, v2), edge_from_pair(v2, v3), edge_from_pair(v3, v0)))
  for pair in list
    e = add_edge!(s, pair...)
    e != ne(s)
    push!(es, e)
  end

  # TODO: Add quad lookup to prevent redundant quads
  push!(s.∂e0, es[1]); push!(s.∂e1, es[2]); push!(s.∂e2, es[3]); push!(s.∂e3, es[4]);
  push!(s.quad_orientation, o)
  push!(s.area, norm(cross(point(s, v1) - point(s, v0), point(s, v3) - point(s, v0))))

  return s
end

# TODO: Take first quad and orient neighbors accordingly
function orient!()
  return nothing
end

quad_edges(s::HasCubicalComplex, q::Int) = SVector(getindex(s.∂e0, q), getindex(s.∂e1, q), getindex(s.∂e2, q), getindex(s.∂e3, q))

# TODO: Make fewer allocations
quad_vertices(s::HasCubicalComplex, q::Int) = get(s.quad_vertex_lookup, q, SVector(0,0,0,0))

quad_area(s, q::Int) = getindex(s.area, q)
quad_orient(s::HasCubicalComplex, e::Int) = getindex(s.quad_orientation, e)

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

    # TODO: Which is src and tgt per Comby?
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
    e0, e1, e2, e3 = quad_edges(s, q)

    I[idx] = q
    I[idx + 1] = q
    I[idx + 2] = q
    I[idx + 3] = q

    J[idx] = e0
    J[idx + 1] = e1
    J[idx + 2] = e2
    J[idx + 3] = e3

    # TODO: This needs some sense of face orientation (assume edges oriented low to high)
    V[idx] = 1
    V[idx + 1] = -1
    V[idx + 2] = 1
    V[idx + 3] = 1
  end

  return sparse(I, J, V)
end
