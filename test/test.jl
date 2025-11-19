using CombinatorialSpaces, Meshes

p = PolyArea((0,0), (2,0), (2,2), (1,3), (0,2))
mesh = discretize(p, DelaunayTriangulation())

EmbeddedDeltaSet2D(mesh)
