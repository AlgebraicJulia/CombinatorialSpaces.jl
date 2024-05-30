module SimplicialComplexes

function add_0_cells(d::HasDeltaSet, t::Trie{Int, Int})
    for v in vertices(d_set)
        t[[v]] = v
    end
end

function add_1_cells(d::HasDeltaSet, t::Trie{Int, Int})
    for e in edges(d)
        v1, v2 = src(d, e), tgt(d, e)
        v1 < v2 || error("Degenerate or unsorted edge: $e")
        haskey(t, [v1, v2]) && error("Duplicate edge: $e")
        t[[v1, v2]] = e
    end
end

function add_2_cells(d::HasDeltaSet, t::Trie{Int, Int})
    for tr in triangles(d)
        v1, v2, v3 = triangle_vertices(d, t)
        v1 < v2 < v3 || error("Degenerate or unsorted trangle: $tr")
        haskey(t, [v1, v2, v3]) && error("Duplicate triangle: $tr")
        t[[v1, v2, v3]] = tr
    end
end

struct SimplicialComplex{D}
    delta_set::D
    complexes::Trie{Int, Int}

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

end
