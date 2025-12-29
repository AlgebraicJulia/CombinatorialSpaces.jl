export CubicalComplex, EmbeddedCubicalComplex2D

using StaticArrays: @SVector, SVector
using GeometryBasics: Point2d, Point3d, QuadFace
using LinearAlgebra: norm, diagm, diag
using SparseArrays

using KernelAbstractions

import Base.show

using GeometryBasics
import GeometryBasics.Mesh

using Makie
import Makie: convert_arguments

include("CubicalCode/CubicalMesh.jl")
include("CubicalCode/CubicalPlotting.jl")
include("CubicalCode/CubicalMatrixDEC.jl")
include("CubicalCode/CubicalKernelDEC.jl")
include("CubicalCode/CubicalPeriodic.jl")
