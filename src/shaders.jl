function get_rasterizer_shaders_code(;
    differential_bary::Bool = false, enable_z_modify::Bool = false,
)
    # TODO allow true
    @assert !differential_bary
    @assert !enable_z_modify

    vertex_shader_code = """
    #version 330 core
    #extension GL_ARB_shader_draw_parameters : enable

    layout(location = 0) in vec4 in_position;

    out int v_layer;
    out int v_offset;

    void main() {
        gl_Position = in_position;
        v_layer = gl_DrawIDARB;
        v_offset = gl_BaseInstanceARB;
    }
    """

    if differential_bary
        @assert false
    else
        geometry_shader_code = """
        #version 330 core

        layout(triangles) in;
        layout(triangle_strip, max_vertices=3) out;

        in int v_layer[];
        in int v_offset[];

        out vec4 var_uvzw;

        void main() {
            int draw_id = v_layer[0];
            int prim_id = gl_PrimitiveIDIn + v_offset[0];

            gl_Layer = draw_id;
            gl_PrimitiveID = prim_id;
            gl_Position = gl_in[0].gl_Position;
            var_uvzw = vec4(1.f, 0.f, gl_in[0].gl_Position.z, gl_in[0].gl_Position.w);
            EmitVertex();

            gl_Layer = draw_id;
            gl_PrimitiveID = prim_id;
            gl_Position = gl_in[1].gl_Position;
            var_uvzw = vec4(0.f, 1.f, gl_in[1].gl_Position.z, gl_in[1].gl_Position.w);
            EmitVertex();

            gl_Layer = draw_id;
            gl_PrimitiveID = prim_id;
            gl_Position = gl_in[2].gl_Position;
            var_uvzw = vec4(0.f, 0.f, gl_in[2].gl_Position.z, gl_in[2].gl_Position.w);
            EmitVertex();
        }
        """
        fragment_shader_code = """
        #version 430
        in vec4 var_uvzw;
        layout(location = 0) out vec4 out_rasterizer;
        void main() {
            out_rasterizer = vec4(
                var_uvzw.x, var_uvzw.y, var_uvzw.z / var_uvzw.w,
                float(gl_PrimitiveID + 1));
        }
        """
    end
    vertex_shader_code, geometry_shader_code, fragment_shader_code
end
