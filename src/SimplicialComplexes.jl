"""
The category of simplicial complexes and Kleisli maps for the convex space monad.
"""
module SimplicialComplexes
export SimplicialComplex, VertexList, has_simplex, GeometricPoint, has_point, has_span, GeometricMap, nv, as_matrix, compose, id, cocenter, primal_vertices, subdivision_maps
using ..Tries
using ..SimplicialSets, ..DiscreteExteriorCalculus
import ACSets: incident, subpart
import AlgebraicInterfaces: dom,codom,compose,id 
import Base:*
import StaticArrays: MVector
import LinearAlgebra: I
import ..DiscreteExteriorCalculus: Barycenter, AbstractDeltaDualComplex
import ..DiscreteExteriorCalculus: PrimalVectorField
#import ..SimplicialSets: nv,ne

function add_0_cells(d::HasDeltaSet, t::Trie{Int, Int})
    for v in vertices(d)
        t[[v]] = v
    end
end

function add_1_cells(d::HasDeltaSet, t::Trie{Int, Int})
    for e in edges(d)
        vs = sort([src(d, e), tgt(d, e)])
        allunique(vs) || error("Degenerate edge: $e")
        haskey(t, vs) && error("Duplicate edge: $e")
        t[vs] = e
    end
end

function add_2_cells(d::HasDeltaSet, t::Trie{Int, Int})
    for tr in triangles(d)
        vs = sort(triangle_vertices(d, tr))
        allunique(vs) || error("Degenerate triangle: $tr") 
        haskey(t, vs) && error("Duplicate triangle: $tr")
        t[vs] = tr
    end
end

struct SimplicialComplex{D}
  delta_set::D
  cache::Trie{Int,Int}

  function SimplicialComplex(d::DeltaSet0D)
    t = Trie{Int,Int}()
    add_0_cells(d, t)
    new{DeltaSet0D}(d, t)
  end

  function SimplicialComplex(d::D) where {D<:AbstractDeltaSet1D}
    t = Trie{Int,Int}()
    add_0_cells(d, t)
    add_1_cells(d, t)
    new{D}(d, t)
  end

  function SimplicialComplex(d::D) where {D<:AbstractDeltaSet2D}
    t = Trie{Int,Int}()
    add_0_cells(d, t)
    add_1_cells(d, t)
    add_2_cells(d, t)
    new{D}(d, t)
  end

  #XX Make this work for oriented types, maybe error for embedded types
  """
  Build a simplicial complex without a pre-existing delta-set.

  In this case any initial values in the trie are meaningless and will be overwritten.
  If you apply this to the cache of a simplicial complex, you may get a non-isomorphic
  Δ-set, but it will be isomorphic as a simplicial complex (i.e. after symmetrizing.)
  `simplices[1]`` is sorted just for predictability of the output--this guarantees that 
  the result will have the same indexing for the vertices in its `cache` as in its
  `delta_set`.
  """
  function SimplicialComplex(dt::Type{D}, t::Trie{Int,Int}) where {D<:HasDeltaSet}
    n = dimension(D)
    simplices = MVector{n + 1,Vector{Vector{Int}}}(fill([], n + 1))
    for k in keys(t)
      push!(simplices[length(k)], k)
    end
    d = D()
    for v in sort(simplices[1])
      t[v] = add_vertex!(d)
    end
    n > 0 && for e in simplices[2]
      t[e] = add_edge!(d, t[e[1]], t[e[2]])
    end
    n > 1 && for tri in simplices[3]
      t[tri] = glue_triangle!(d, t[tri[1]], t[tri[2]], t[tri[3]])
    end
    n > 2 && for tet in simplices[4]
      t[tet] = glue_tetrahedron!(d, t[tet[1]], t[tet[2]], t[tet[3]], t[tet[4]])
    end
    new{D}(d, t)
  end
end

#XX: Should this output something oriented?
"""
Build a simplicial complex from a trie, constructing a delta-set of the minimal
dimension consistent with the trie.
"""
SimplicialComplex(t::Trie{Int,Int}) = SimplicialComplex(DeltaSet(max(height(t)-1,0)),t) 

for f in [:nv,:ne,:ntriangles,:dimension] 
    @eval SimplicialSets.$f(sc::SimplicialComplex) = $f(sc.delta_set)
end

struct VertexList #XX parameterize by n? 
    vs::Vector{Int} # must be sorted
    function VertexList(vs::Vector{Int}; sorted=false)
        new(sorted ? vs : sort(vs))
    end
    function VertexList(d::HasDeltaSet, s::Simplex{n,0}) where n
        new(sort(simplex_vertices(d,s)))
    end
end

"""
Iterator over proper subsimplices of a simplex in reversed binary order.

Example
```julia-repl
julia> vl = VertexList([1,4,9])
VertexList([1, 4, 9])
julia> iter = SubsimplexIterator(vl)
SubsimplexIterator(VertexList([1, 4, 9]), 7)
julia> collect(iter)
7-element Vector{VertexList}:
 VertexList(Int64[])
 VertexList([1])
 VertexList([4])
 VertexList([1, 4])
 VertexList([9])
 VertexList([1, 9])
 VertexList([4, 9])
```
"""
struct SubsimplexIterator
  vl::VertexList
  length::Int
  #Note that an n-simplex has 2^(n+1)-1 subsimplices, with n+1 vertices.
  SubsimplexIterator(vl::VertexList) = new(vl, 2^length(vl.vs)-1)
end
Base.length(iter::SubsimplexIterator) = iter.length
Base.eltype(iter::SubsimplexIterator) = VertexList
function Base.iterate(iter::SubsimplexIterator,i=0)
    if i >= iter.length
        return nothing
    end
    ds = digits(i,base=2)
    mask = Bool[ds;fill(0,length(iter.vl.vs)-length(ds))]
    (VertexList(iter.vl.vs[mask]),i+1)
end

Base.length(s::VertexList) = length(s.vs)
Base.lastindex(s::VertexList) = lastindex(s.vs)
has_simplex(sc::SimplicialComplex,s::VertexList) = haskey(sc.cache, s.vs)

Base.getindex(v::VertexList, i) = v.vs[i]

function Base.getindex(sc::SimplicialComplex, s::VertexList)::Simplex
    has_simplex(sc,s) || error("Simplex not found: $s")
    Simplex{length(s)}(sc.cache[s.vs])
end

function Base.union(vs1::VertexList, vs2::VertexList)
    out = Int[]
    i, j = 1, 1
    while (i <= length(vs1)) && (j <= length(vs2))
        v1, v2 = vs1[i], vs2[j]
        if (v1 == v2)
            push!(out, v1)
            i += 1
            j += 1
        elseif (v1 <= v2)
            push!(out, v1)
            i += 1
        else
            push!(out, v2)
            j += 1
        end
    end
    if (i <= length(vs1))
        append!(out, vs1[i:end])
    end
    if (j <= length(vs2))
        append!(out, vs2[j:end])
    end
    VertexList(out, sorted=true)
end


#XX: This type is maybe more trouble than it's worth?
#yeah kill
"""
A point in an unspecified simplicial complex, given by its barycentric coordinates.
Constructed via a dense vector of coordinates.
"""
struct GeometricPoint
  bcs::Vector{Float64} #XX: Need a sparse form?
  function GeometricPoint(bcs, checked=true)
    if checked
      sum(bcs) ≈ 1 || error("Barycentric coordinates must sum to 1")
      all(x -> 1 ≥ x ≥ 0, bcs) || error("Barycentric coordinates must be between 0 and 1")
    end
    new(bcs)
  end
end
Base.show(p::GeometricPoint) = "GeometricPoint($(p.bcs))"
coords(p::GeometricPoint)=p.bcs

"""
A simplicial complex contains a geometric point if and only if it contains the combinatorial simplex spanned
by the vertices wrt which the point has a nonzero coordinate.
"""
has_point(sc::SimplicialComplex, p::GeometricPoint) = has_simplex(sc, VertexList(findall(x->x>0,coords(p))))
"""
A simplicial complex contains the geometric simplex spanned by a list of geometric points if and only if it 
contains the combinatorial simplex spanned by all the vertices wrt which some geometric point has a nonzero coordinate.
"""
has_span(sc::SimplicialComplex,ps::Vector{GeometricPoint}) = has_simplex(sc,reduce(union,VertexList.(map(cs->findall(x->x>0,cs),(coords.(ps))))))

#geoemtric map between simplicial complexes, given as a list of geometric points in the codomain 
#indexed by the 0-simplices of the domain.
struct GeometricMap{D,D′}
  dom::SimplicialComplex{D}
  cod::SimplicialComplex{D′}
  points::Vector{GeometricPoint}
  function GeometricMap(sc::SimplicialComplex{D}, sc′::SimplicialComplex{D′}, points::Vector{GeometricPoint};checked::Bool=true) where {D,D′}
    if checked 
      length(points) == nv(sc) || error("Number of points must match number of vertices in domain")
      all(map(x->has_span(sc′,points[x]),keys(sc.cache))) || error("Span of points in simplices of domain must lie in codomain")
    end
    new{D,D′}(sc, sc′, points)
  end
end
GeometricMap(sc::SimplicialComplex{D},sc′::SimplicialComplex{D′},points::AbstractMatrix;checked::Bool=true) where {D,D′} = GeometricMap(sc,sc′,GeometricPoint.(eachcol(points)),checked=checked)
dom(f::GeometricMap) = f.dom
codom(f::GeometricMap) = f.cod
#want f(n) to give values[n]?
"""
Returns the data-centric view of f as a matrix whose i-th column 
is the coordinates of the image of the i-th vertex under f.
"""
as_matrix(f::GeometricMap) = hcat(coords.(f.points)...)
compose(f::GeometricMap, g::GeometricMap) = GeometricMap(f.dom, g.cod, as_matrix(g)*as_matrix(f))
id(sc::SimplicialComplex) = GeometricMap(sc,sc,GeometricPoint.(eachcol(I(nv(sc)))))

#make sparse matrices
function GeometricMap(sc::SimplicialComplex,::Barycenter)
  dom = SimplicialComplex(extract_dual(sc.delta_set))
  #Vertices of dom correspond to vertices, edges, triangles of sc.
  mat = zeros(Float64,nv(sc),nv(dom))
  for i in 1:nv(sc) mat[i,i] = 1 end
  for i in 1:ne(sc) for n in edge_vertices(sc.delta_set,i) mat[n,nv(sc)+i] = 1/2 end end
  for i in 1:ntriangles(sc) for n in triangle_vertices(sc.delta_set,i) mat[n,nv(sc)+ne(sc)+i] = 1/3 end end
  GeometricMap(dom,sc,mat)
end
#accessors for the nonzeros in a column of the matrix

"""
The geometric maps from a deltaset's subdivision to itself and back again.

Warning: the second returned map is not actually a valid geometric map as edges 
of the primal delta set will run over multiple edges of the dual. So, careful composing
with it etc.
"""
function subdivision_maps(primal_s::EmbeddedDeltaSet,alg=Barycenter())
  prim = SimplicialComplex(primal_s)
  s = dualize(primal_s)
  subdivide_duals!(s,alg)
  dual = SimplicialComplex(extract_dual(s))
  mat = zeros(Float64,nv(prim),nv(dual))
  pvs = map(i->primal_vertices(s,i),1:nv(dual))
  weights = 1 ./(length.(pvs))
  for j in 1:nv(dual)
    for v in pvs[j]
      mat[v,j] = weights[j]
    end
  end
  a = GeometricMap(dual,prim,mat)
  mat′ = zeros(Float64,nv(dual),nv(prim))
  for i in 1:nv(prim)
    mat′[subpart(s,i,:vertex_center),i] = 1
  end
  a, GeometricMap(prim,dual,mat′,checked=false)
end

function pullback_primal(f::GeometricMap, v::PrimalVectorField{T}) where T
  nv(f.cod) == length(v) || error("Vector field must have same number of vertices as codomain")
  PrimalVectorField(T.(eachcol(hcat(v.data...)*as_matrix(f))))
end
#Is restriction the transpose?
*(f::GeometricMap,v::PrimalVectorField) = pullback_primal(f,v)

function dual_vertex_dimension(s::AbstractDeltaDualComplex,v::DualV)
  n = v.data
  !isempty(incident(s,n,:vertex_center)) ? 0 :
  !isempty(incident(s,n,:edge_center)) ? 1 :
  !isempty(incident(s,n,:tri_center)) ? 2 : 3
end

simplex_name_dict = Dict(0=>:vertex,1=>:edge,2=>:tri,3=>:tet)

#XX: the parts data structure allowing data to be like whatever is awful
function cocenter(s::AbstractDeltaDualComplex,v::DualV)
  n = dimension(s)
  v = v.data
  for i in 0:n
    inc = incident(s,v,Symbol(simplex_name_dict[i],:(_center)))
    if !isempty(inc)
      return Simplex{i}(only(inc))
    end
  end
end
cocenter(s::AbstractDeltaDualComplex,n::Int) = cocenter(s,DualV(n))
primal_vertices(s::AbstractDeltaDualComplex,v::DualV) = simplex_vertices(s,cocenter(s,v))
primal_vertices(s::AbstractDeltaDualComplex,n::Int) = simplex_vertices(s,cocenter(s,DualV(n)))

#dimension(x::Simplex{n}) where {n} = n

end