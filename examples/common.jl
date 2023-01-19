function get_triangle(dev)
    positions = DiffRast.to_device(dev, Float32[
        -1; -1; 0;;
         1; -1; 0;;
        -1;  1; 0;;;
    ])
    colors = DiffRast.to_device(dev, Float32[
        1; 0; 0;;
        0; 1; 0;;
        0; 0; 1;;;
    ])
    indices = DiffRast.to_device(dev, UInt32[0; 1; 2;;])
    positions, colors, indices
end

function get_plane(dev)
    positions = DiffRast.to_device(dev, Float32[
        -1; -1; 0;;
         1; -1; 0;;
        -1;  1; 0;;
         1;  1; 0;;;
    ])
    colors = DiffRast.to_device(dev, Float32[
        0.251; 0.388; 0.847;; # blue
        0.796; 0.235; 0.2;; # red
        0.22; 0.596; 0.149;; # green
        0.584; 0.345; 0.698;;; # purple
    ])
    indices = DiffRast.to_device(dev, UInt32[
        0; 1; 2;;
        2; 3; 1;;
    ])
    positions, colors, indices
end

function get_cube(dev)
    positions = DiffRast.to_device(dev, Float32[
        -0.5; -0.5; -0.5;;
        -0.5; -0.5;  0.5;;
        -0.5;  0.5; -0.5;;
        -0.5;  0.5;  0.5;;
         0.5; -0.5; -0.5;;
         0.5; -0.5;  0.5;;
         0.5;  0.5; -0.5;;
         0.5;  0.5;  0.5;;;
    ])
    colors = DiffRast.to_device(dev, Float32[
        0; 0; 0;;
        0; 0; 1;;
        0; 1; 0;;
        0; 1; 1;;
        1; 0; 0;;
        1; 0; 1;;
        1; 1; 0;;
        1; 1; 1;;;
    ])
    indices = DiffRast.to_device(dev, UInt32[
        0; 6; 4;;
        0; 2; 6;;
        1; 5; 7;;
        1; 7; 3;;
        0; 3; 2;;
        0; 1; 3;;
        4; 6; 7;;
        4; 7; 5;;
        2; 7; 6;;
        2; 3; 7;;
        0; 4; 5;;
        0; 5; 1;;
    ])
    positions, colors, indices
end

function get_rotation_h(rx, ry, rz)
    r = zeros(Float32, 4, 4)
    r[1:3, 1:3] .= RotXYZ(rx, ry, rz)
    r[4, 4] = 1
    r
end

function get_random_rotation_translation_h(ts)
    r = zeros(Float32, 4, 4)
    r[1:3, 1:3] .= rand(RotXYZ)
    r[1:3, 4] .= rand(Float32, 3) .* ts .- ts * 0.5f0
    r[4, 4] = 1
    r
end

function get_translation_h(tx, ty, tz)
    r = zeros(Float32, 4, 4)
    r[1:3, 1:3] .= SMatrix{3, 3, Float32, 9}(I)
    r[1:4, 4] .= (tx, ty, tz, 1)
    r
end

function load_poses(config_file; with_images::Bool, synthetic::Bool)
    base_dir = dirname(config_file)
    config = JSON.parse(read(config_file, String))

    Rs = SMatrix{3, 3, Float32, 9}[]
    ts = SVector{3, Float32}[]
    images = Array{UInt8, 3}[]

    function load_image(file_path)
        raw = Float32.(channelview(RGB{Float32}.(rotr90(load(file_path)))))
        round.(UInt8, raw .* 255f0)
    end

    for (j, frame_config) in enumerate(config["frames"])
        P = Float32.(hcat(frame_config["transform_matrix"][1:(end - 1)]...)')
        R = SMatrix{3, 3, Float32, 9}(@view(P[1:3, 1:3]))
        t = SVector{3, Float32}(@view(P[1:3, 4]))
        push!(Rs, R)
        push!(ts, t)

        if with_images
            file_path = joinpath(base_dir, frame_config["file_path"])
            @show j, file_path
            synthetic && (file_path *= ".png";)
            push!(images, load_image(file_path))
        end
    end
    Rs, ts, images
end

function to_rt(r, t)
    Array(SMatrix{4, 4, Float32, 16}(
        r[1, 1], r[2, 1], r[3, 1], 0f0,
        r[1, 2], r[2, 2], r[3, 2], 0f0,
        r[1, 3], r[2, 3], r[3, 3], 0f0,
        t[1], t[2], t[3], 1f0))
end

function show_rot(r)
    r1 = RotXYZ(r)
    a1, a2, a3 = rad2deg(r1.theta1), rad2deg(r1.theta2), rad2deg(r1.theta3)
    @show a1, a2, a3
end
