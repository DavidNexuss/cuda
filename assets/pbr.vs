#version 330 core
layout(location = 0)  in vec3 a_Position;
layout(location = 1)  in vec3 a_Normal;
layout(location = 2)  in vec2 a_UV;

uniform mat4 u_ViewMat;
uniform mat4 u_ProjMat;
uniform mat4 u_WorldMat;

out vec2 f_uv;
out vec4 f_pos;
out vec3 f_normal;
void main() { 

    f_pos = u_WorldMat * vec4(a_Position, 1.0);
    gl_Position = u_ProjMat * u_ViewMat * f_pos;
    f_uv = f_pos.xz;
    f_normal = a_Normal;
}
