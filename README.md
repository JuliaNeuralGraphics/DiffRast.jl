# DiffRast.jl

A differentiable rasterizer

## Examples

1. Render primitive at random pose and save the results.

```bash
julia --threads=auto --project=. examples/primitive.jl
```

|Antialiased|Render|UV|Triangle ID|
|-|-|-|-|
|![image](/data/antialiased.png)|![image](/data/interpolation.png)|![image](/data/uv.png)|![image](/data/triangle-ids.png)|

2. Learn vertex positions and colors of a randomly initialized cube.

```bash
julia --threads=auto --project=. examples/cube.jl
```

https://user-images.githubusercontent.com/17990405/191010283-21cd8c0d-69e6-455b-b30b-49156e485c62.mp4

## Notes

- Julia 1.8+ is needed.
- Supports only instance mode for now.
- Batch matrix multiplication is done using `NNlib.⊠` operator, for AMDGPU we'd need to add that.
rocBLAS should have the necessary stuff.
