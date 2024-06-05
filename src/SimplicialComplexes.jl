module SimplicialComplexes
export SimplicialComplex, VertexList, has_simplex
using ..Tries
using ..SimplicialSets

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
    cache::Trie{Int, Int}

    function SimplicialComplex(d::D) where {D<:AbstractDeltaSet1D}
        t = Trie{Int, Int}()
        add_0_cells(d, t)
        add_1_cells(d, t)
        new{D}(d, t)
    end

    function SimplicialComplex(d::D) where {D<:AbstractDeltaSet2D}
        t = Trie{Int, Int}()
        add_0_cells(d, t)
        add_1_cells(d, t)
        add_2_cells(d, t)
        new{D}(d, t)
    end
end

struct VertexList #XX parameterize by n? remember to replace sort! w sort
    vs::Vector{Int} # must be sorted
    function VertexList(vs::Vector{Int}; sorted=false)
        new(sorted ? vs : sort(vs))
    end
    function VertexList(d::HasDeltaSet, s::Simplex{n,0}) where n
        new(sort(simplex_vertices(d,s)))
    end
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

#TODO: get a point by barycentric coordinates, maps
end