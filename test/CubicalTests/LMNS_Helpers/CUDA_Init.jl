const HAS_CUDA = Base.find_package("CUDA") !== nothing

if HAS_CUDA
  import CUDA
  import CUDA.CUSPARSE

  CUDA.allowscalar(false)

  const USE_CUDA = CUDA.functional()
  println("CUDA is functional: $USE_CUDA")

  to_device(arr::AbstractVector{T}) where T = USE_CUDA ? CUDA.CuVector{T}(arr) : arr
  to_device(mat::AbstractMatrix{T}) where T = USE_CUDA ? CUDA.CUSPARSE.CuSparseMatrixCSC{T}(mat) : SparseMatrixCSC{T}(mat)
else
  const USE_CUDA = false
  println("CUDA package not found. Running on CPU.")

  to_device(arr::AbstractVector{T}) where T = arr
  to_device(mat::AbstractMatrix{T}) where T = SparseMatrixCSC{T}(mat)
end
