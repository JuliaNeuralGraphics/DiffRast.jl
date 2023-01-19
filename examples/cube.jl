using CUDAKernels
using DiffRast
using FileIO
using GL
using ImageCore
using ImageIO
using LinearAlgebra
using Rotations
using Optimisers
using StaticArrays
using Zygote
using VideoIO

include("common.jl")

x = (xor.((0:254), (0:254)'))
tex = colorview(Gray, Gray.(x/255))

function compose_frame(y, ŷ)
    y_img = colorview(RGB{N0f8}, round.(UInt8, clamp01!(y[:, :, :, 1]) * 255))
    ŷ_img = colorview(RGB{N0f8}, round.(UInt8, clamp01!(ŷ[:, :, :, 1]) * 255))
    hcat(rotl90(y_img), rotl90(ŷ_img))
end

function train_step!(rasterizer, parameters, indices, optimizer_state, ŷ, mvp)
    ∇ = Zygote.gradient(parameters) do (p, c)
        _, _, y = DiffRast.ad_render(p, c; rasterizer, indices, mvp)
        sum((y .- ŷ).^2) / length(y)
    end
    Optimisers.update!(optimizer_state, parameters, ∇[1])
end

function main()
    dev = CUDADevice()
    positions, colors, indices = get_cube(dev)

    ts = 0.1f0
    ξ = rand(dev, Float32, size(positions)) .* ts .- ts * 0.5f0
    positions_opt = positions .+ ξ
    colors_opt = copy(colors)
    parameters = (positions_opt, colors_opt)

    width, height = 1024, 1024
    ctx = GL.Context("DiffRast"; width, height)
    rasterizer = DiffRast.Rasterizer(; width, height)

    aspect_ratio = Float32(width) / Float32(height)
    projection = DiffRast.to_device(dev, GL.perspective(45f0, aspect_ratio, 0.1, 100))
    view_m = DiffRast.to_device(dev, get_translation_h(-0.15, 0, -5))
    model_m = DiffRast.to_device(dev, get_rotation_h(π / 16f0, π / 16f0, π / 16f0))
    vp = projection * view_m
    mvp = vp * model_m

    _, _, ŷ = DiffRast.render(positions, colors; rasterizer, indices, mvp)

    video_file = "cube.mp4"
    writer = open_video_out(
        video_file, zeros(RGB{N0f8}, height, width * 2);
        framerate=120, target_pix_fmt=VideoIO.AV_PIX_FMT_YUV420P)

    optimizer = Adam(3f-4)
    optimizer_state = Optimisers.setup(optimizer, parameters)
    for i in 1:2000
        optimizer_state, parameters = train_step!(
            rasterizer, parameters, indices, optimizer_state, ŷ, vp)

        if i % 1 == 0
            _, _, y = DiffRast.render(positions_opt, colors_opt; rasterizer, indices, mvp)
            write(writer, compose_frame(Array(y), Array(ŷ)))

            loss = sum((y .- ŷ).^2 / length(y))
            println("$i | Loss: $loss")
        end
    end

    close_video_out!(writer)
    GL.delete!(rasterizer)
    GL.delete!(ctx)
end
main()
