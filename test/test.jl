using CombinatorialSpaces, Meshes

p = PolyArea((0,0), (2,0), (2,2), (1,3), (0,2))
mesh = discretize(p, DelaunayTriangulation())

EmbeddedDeltaSet2D(mesh)

points = [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)]
connec = connect.([(1,4,3,2),(5,6,7,8),(1,2,6,5),(3,4,8,7),(1,5,8,4),(2,3,7,6)])
mesh   = SimpleMesh(points, connec)
ref = refine(mesh, TriSubdivision())

EmbeddedDeltaSet2D(ref)
