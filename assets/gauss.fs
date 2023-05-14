#version 330 core
  
in vec2 uv;
out vec3 color;

uniform sampler2D u_input;
uniform bool u_horizontal;
uniform float u_weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main()
{             
    vec2 tex_offset = 1.0 / textureSize(u_input, 0); // gets size of single texel
    vec3 result = texture(u_input, uv).rgb * u_weight[0]; // current fragment's contribution
    if(u_horizontal)
    {
        for(int i = 1; i < 5; ++i)
        {
            result += texture(u_input, uv + vec2(tex_offset.x * i, 0.0)).rgb * u_weight[i];
            result += texture(u_input, uv - vec2(tex_offset.x * i, 0.0)).rgb * u_weight[i];
        }
    }
    else
    {
        for(int i = 1; i < 5; ++i)
        {
            result += texture(u_input, uv + vec2(0.0, tex_offset.y * i)).rgb * u_weight[i];
            result += texture(u_input, uv - vec2(0.0, tex_offset.y * i)).rgb * u_weight[i];
        }
    }
    color = result;
}
