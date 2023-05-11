#version 330 core
layout(location = 0)  in vec3 aPosition;
layout(location = 1)  in vec3 aNormal;
layout(location = 2)  in vec2 aUV;

uniform mat4 uViewMat;
uniform mat4 uProjMat;
uniform mat4 uWorldMat;

out vec2 uv;
out vec4 pos;
void main() { 

    pos = uWorldMat * vec4(aPosition, 1.0);
    gl_Position = uProjMat * uViewMat * pos;
    uv = aPosition.xz * 1000;
}
