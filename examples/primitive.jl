using CUDAKernels
using DiffRast
using FileIO
using GL
using ImageCore
using ImageIO
using KernelAbstractions
using LinearAlgebra
using Rotations
using StaticArrays

include("common.jl")

function main()
    dev = CUDADevice()
    # positions, colors, indices = get_triangle(dev)
    # positions, colors, indices = get_plane(dev)
    positions, colors, indices = get_cube(dev)

    width, height = 512, 512
    ctx = GL.Context("DiffRast"; width, height)
    rasterizer = DiffRast.Rasterizer(; width, height)

    aspect_ratio = Float32(width) / Float32(height)
    projection = DiffRast.to_device(dev, GL.perspective(45f0, aspect_ratio, 0.1, 100))
    view_m = DiffRast.to_device(dev, get_translation_h(-0.15, 0, -5))
    model_m = DiffRast.to_device(dev, get_rotation_h(π / 16f0, π / 16f0, π / 16f0))
    mvp = projection * view_m * model_m
    rasterization, interpolations, antialiased = DiffRast.render(
        positions, colors; rasterizer, indices, mvp)

    # Save results.
    rasterization_host = Array(rasterization)
    interpolations_host = Array(interpolations)
    antialiased_host = Array(antialiased)

    uvs = vcat(
        rasterization_host[1:2, :, :, 1],
        zeros(Float32, 1, size(rasterization, 2), size(rasterization, 3)))
    triangle_ids = rasterization_host[4, :, :, 1]
    triangle_ids ./= maximum(triangle_ids)

    save("uv.png", rotl90(colorview(RGB{Float32}, uvs)))
    save("triangle-ids.png", rotl90(colorview(Gray{Float32}, triangle_ids)))
    save("interpolation.png", rotl90(colorview(
        RGB{Float32}, interpolations_host[:, :, :, 1])))
    save("antialiased.png", rotl90(colorview(
        RGB{Float32}, antialiased_host[:, :, :, 1])))

    GL.delete!(rasterizer)
    GL.delete!(ctx)
end
main()
