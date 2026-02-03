using Test
using CombinatorialSpaces
using GeometryBasics: Point2, Point3
using Catlab
using Catlab.CategoricalAlgebra: ACSetCategory, ACSetCat
using Catlab.CategoricalAlgebra.Subobjects: Subobject, negate, non, meet, join
using Catlab.Theories: @withmodel

@testset "Mesh Decomposition with Catlab 0.17" begin
  @testset "2D Mesh Quadrants" begin
    # Create a simple 2D triangulated grid
    s = triangulated_grid(10, 10, 5, 5, Point3{Float64})
    
    # Test category creation with ACSetCat
    @test_nowarn ACSetCategory(ACSetCat(s))
    𝒞 = ACSetCategory(ACSetCat(s))
    
    # Define quadrant partition function
    quadrants(x) = Int(x[1] > 5) + 2*Int(x[2] > 5)
    
    # Create cover mesh function
    function cover_mesh(partition_function, s, cat)
      vertex_partition = map(partition_function, s[:point])
      parts = map(unique(vertex_partition)) do p
        vp = findall(i->i==p, vertex_partition)
        subobj = Subobject(s; V=vp)
        @withmodel cat (negate, non) begin
          sp = non(negate(subobj))
        end
      end
      return parts
    end
    
    # Test cover mesh creation
    @test_nowarn cover_mesh(quadrants, s, 𝒞)
    quads = cover_mesh(quadrants, s, 𝒞)
    
    # Test that we got 4 quadrants
    @test length(quads) == 4
    
    # Test that each element is a Subobject
    @test all(q -> q isa Subobject, quads)
    
    # Test meet operation
    @test_nowarn @withmodel 𝒞 (meet,) begin
      meet(quads[1], quads[2])
    end
    
    # Test join operation
    @test_nowarn @withmodel 𝒞 (join,) begin
      join(quads[1], quads[3])
    end
    
    # Test nerve construction
    function nerve(cover::Vector{T}, cat) where T <: Subobject
      n = length(cover)
      @withmodel cat (meet,) begin
        map(1:n) do i
          map(i:n) do j
            ui, uj = cover[i], cover[j]
            uij = meet(ui, uj)
            (uij, i, j)
          end
        end |> Iterators.flatten
      end
    end
    
    @test_nowarn nerve(quads, 𝒞)
    D = nerve(quads, 𝒞)
    
    # Test that nerve returns the correct number of intersections
    # For n=4 quadrants, we should have 4 + 3 + 2 + 1 = 10 pairs
    @test length(collect(D)) == 10
  end
  
  @testset "1D Circle Mesh" begin
    # Create a circular mesh
    function circle(n, c)
      mesh = EmbeddedDeltaSet1D{Bool, Point2{Float64}}()
      map(range(0, 2π - (π/(2^(n-1))); step=π/(2^(n-1)))) do t
        add_vertex!(mesh, point=Point2(cos(t), sin(t))*(c/(2π)))
      end
      add_edges!(mesh, 1:(nv(mesh)-1), 2:nv(mesh))
      add_edge!(mesh, nv(mesh), 1)
      dualmesh = EmbeddedDeltaDualComplex1D{Bool, Float64, Point2{Float64}}(mesh)
      subdivide_duals!(dualmesh, Circumcenter())
      mesh, dualmesh
    end
    
    mesh, dualmesh = circle(6, 100)
    
    # Test category creation for 1D mesh
    @test_nowarn ACSetCategory(ACSetCat(dualmesh))
    𝒞₁ = ACSetCategory(ACSetCat(dualmesh))
    
    # Define pizza slice partition
    pizza_slices(x) = Int(x[1] > 0) + 2*Int(x[2] > 0)
    
    # Create cover mesh function (same as above)
    function cover_mesh_1d(partition_function, s, cat)
      vertex_partition = map(partition_function, s[:point])
      parts = map(unique(vertex_partition)) do p
        vp = findall(i->i==p, vertex_partition)
        subobj = Subobject(s; V=vp)
        @withmodel cat (negate, non) begin
          sp = non(negate(subobj))
        end
      end
      return parts
    end
    
    # Test cover mesh creation for circle
    @test_nowarn cover_mesh_1d(pizza_slices, dualmesh, 𝒞₁)
    circ_quads = cover_mesh_1d(pizza_slices, dualmesh, 𝒞₁)
    
    # Test that we got quadrants (should be 4)
    @test length(circ_quads) == 4
    
    # Test that each element is a Subobject
    @test all(q -> q isa Subobject, circ_quads)
  end
  
  @testset "NerveCover Type" begin
    s = triangulated_grid(10, 10, 5, 5, Point3{Float64})
    𝒞 = ACSetCategory(ACSetCat(s))
    
    quadrants(x) = Int(x[1] > 5) + 2*Int(x[2] > 5)
    
    function cover_mesh(partition_function, s, cat)
      vertex_partition = map(partition_function, s[:point])
      parts = map(unique(vertex_partition)) do p
        vp = findall(i->i==p, vertex_partition)
        subobj = Subobject(s; V=vp)
        @withmodel cat (negate, non) begin
          sp = non(negate(subobj))
        end
      end
      return parts
    end
    
    quads = cover_mesh(quadrants, s, 𝒞)
    
    # Define NerveCover type
    import Catlab.Sheaves: AbstractCover
    
    struct NerveCover{T, X, C} <: AbstractCover
      vertices::Dict{T, Int}
      basis::Vector{X}
      cat::C
    end
    
    function NerveCover(subobjects::Vector{X}, cat) where X <: Subobject
      lookup = enumerate(subobjects)
      vertices = Dict{Int, Int}(i=>i for (i, _) in lookup)
      return NerveCover{Int, Subobject, typeof(cat)}(vertices, subobjects, cat)
    end
    
    function Base.getindex(K::NerveCover, I::Vararg{Int})
      @withmodel K.cat (meet,) begin
        map(I) do i
          K.basis[i]
        end |> x->foldl(meet, x)
      end
    end
    
    # Test NerveCover construction
    @test_nowarn NerveCover(quads, 𝒞)
    K = NerveCover(quads, 𝒞)
    
    # Test length
    @test length(K.basis) == 4
    
    # Test indexing with meet
    @test_nowarn K[1, 2]
    result = K[1, 2]
    @test result isa Subobject
  end
end
