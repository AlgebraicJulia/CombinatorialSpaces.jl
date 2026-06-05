module CombinatorialSpacesTetGenExt

using CombinatorialSpaces
using CombinatorialSpaces.CombMeshes
using GeometryBasics: Mesh, MetaMesh, Point, Point3, Point3d, QuadFace
using TetGen: tetrahedralize

function CombMeshes.parallelepiped(;lx::Real = 1.0, ly::Real = 1.0, lz::Real = 1.0, dx::Real = 0.0, dy::Real = 0.0, point_type::Type{<:Point3} = Point3d, tetcmd::String = "pQq1.414a0.1")
	points = point_type[
    (0.0, 0.0, 0.0), (dx, dy, lz), (0.0, ly, 0.0), (dx, ly+dy, lz),
    (lx, 0.0, 0.0), (lx+dx, dy, lz), (lx, ly, 0.0), (lx+dx, ly+dy, lz)]

  faces = QuadFace{Cint}[
    [1,2,4,3], [5,6,8,7], [1,2,6,5],
    [3,4,8,7], [1,3,7,5], [2,4,8,6]]

  tet_mesh = tetrahedralize(Mesh(points, faces), tetcmd)

  s = EmbeddedDeltaSet3D(tet_mesh)

  orient!(s)
  s[:edge_orientation] = false
  s[:tri_orientation] = false

  return s
end

function CombMeshes.tetgen_readme_mesh()
  points = Point{3, Float64}[
    (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
    (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    (0.0, 0.0, 12.0), (2.0, 0.0, 12.0),
    (2.0, 2.0, 12.0), (0.0, 2.0, 12.0)]
  facets = QuadFace{Cint}[
    1:4, 5:8,
    [1,5,6,2],
    [2,6,7,3],
    [3, 7, 8, 4],
    [4, 8, 5, 1]]
  markers = Cint[-1, -2, 0, 0, 0, 0]
  msh = MetaMesh(points, facets; markers)
  tet_msh = tetrahedralize(msh, "Qvpq1.414a0.1")
  s = EmbeddedDeltaSet3D(tet_msh)
  orient!(s)
  s[:edge_orientation] = false
  s[:tri_orientation] = false
  s
end

end
