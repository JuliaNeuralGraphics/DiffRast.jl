function load_model(model_file::String)
    endswith(model_file, ".obj") || error(
        "Only .obj files are supported. But given file `$model_file`.")

    vertices = SVector{3, Float32}[]
    colors = SVector{3, Float32}[]
    indices = SVector{3, UInt32}[]

    file = open(model_file, "r")
    for line in eachline(file)
        line = strip(chomp(line))
        isascii(line) || error("Non-ascii is not supported, line: `$line`.")
        (startswith(line, "#") || isempty(line) || all(iscntrl, line)) && continue

        line_elements = split(line)
        primitive = popfirst!(line_elements)

        if primitive == "v"
            (length(line_elements) != 3 && length(line_elements) != 6) && error(
                "Unsupported line format `$line`.\n" *
                "Supported format: `v x y z [r g b]`.\n" *
                "Parsed line to: $line_elements.")

            has_colors = length(line_elements) == 6
            parsed_elements = parse.(Float32, line_elements)
            push!(vertices, SVector{3, Float32}(parsed_elements[1:3]))
            has_colors && push!(colors, SVector{3, Float32}(parsed_elements[4:6]))
        elseif primitive == "f"
            length(line_elements) != 3 && error(
                "Unsupported line format `$line`.\n" *
                "Supported format: `f i1 i2 i3`.\n" *
                "Parsed line to: $line_elements.")

            parsed_elements = parse.(UInt32, line_elements)
            @assert !any(parsed_elements .== 0)
            @assert !any(parsed_elements .< 0)
            push!(indices, SVector{3, UInt32}(parsed_elements .- 1))
        end
    end
    close(file)

    length(colors) > 0 && (@assert length(colors) == length(vertices);)
    (
        reshape(reinterpret(Float32, vertices), 3, length(vertices), 1),
        reshape(reinterpret(Float32, colors),  3, length(colors), 1),
        reshape(reinterpret(UInt32, indices), 3, length(indices)))
end

function check_shape(positions, colors, indices)
    (ndims(positions) != 3 || size(positions, 1) != 3) && error(
        "`positions` must be of `3xNxB` shape, instead it is `$(size(positions)).`")
    (ndims(colors) != 3 || size(colors, 1) != 3) && error(
        "`colors` must be of `3xNxB` shape, instead it is `$(size(colors)).`")
    _, pn, pb = size(positions)
    _, cn, cb = size(colors)
    (pn != cn || pb != cb) && error(
        "Number of positions and colors for them must be the same. " *
        "But is instead `$(size(positions))` vs `$(size(colors))`.")
    (ndims(indices) != 2 || size(indices, 1) != 3) && error(
        "`indices` must be of `3xK` shape, instead it is `$(size(indices))`.")
end
ChainRulesCore.@non_differentiable check_shape(::Any...)

function render(
    positions::P, colors; rasterizer::Rasterizer, indices, mvp,
) where P
    check_shape(positions, colors, indices)

    dev = DiffRast.device_from_type(P)
    n_vertices, n = size(positions, 2), size(positions, 3)

    v_pad = ones(dev, Float32, (1, n_vertices, n))
    clip_positions = mvp ⊠ vcat(positions, v_pad)

    rasterization = rasterizer(clip_positions; indices)
    interpolations = DiffRast.interpolate(colors, rasterization; indices)
    antialiased, _, _ = DiffRast.antialias(
        interpolations, clip_positions; rasterization, indices)

    rasterization, interpolations, antialiased
end

function ad_render(
    positions::P, colors; rasterizer::Rasterizer, indices, mvp,
) where P
    check_shape(positions, colors, indices)

    dev = DiffRast.device_from_type(P)
    n_vertices, n = size(positions, 2), size(positions, 3)

    v_pad = ones(dev, Float32, (1, n_vertices, n))
    clip_positions = mvp ⊠ vcat(positions, v_pad)

    rasterization = rasterizer(clip_positions; indices)
    interpolations = DiffRast.interpolate(colors, rasterization; indices)
    antialiased = DiffRast.antialias(
        interpolations, clip_positions; rasterization, indices)

    rasterization, interpolations, antialiased
end

function apply_tid_mask!(target, rasterization::R) where R
    dev = device_from_type(R)
    wait(apply_mask_kernel!(dev)(
        target, rasterization; ndrange=size(rasterization)[2:3]))
    target
end
ChainRulesCore.@non_differentiable apply_tid_mask!(::Any...)

@kernel function apply_mask_kernel!(target, rasterization)
    x, y = @index(Global, NTuple)
    if rasterization[4, x, y, 1] == 0
        for i in 1:3
            target[i, x, y] = 0
        end
    end
end
