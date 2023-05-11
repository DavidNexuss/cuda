#version 330 core

uniform sampler2D envMap;
uniform sampler2D diffuseTexture;
uniform vec3 ro;
uniform vec3 rd;
uniform int isBack;

in vec4 pos;
in vec2 uv;
out vec3 color;

#define M_PI 3.1415

vec2 envUV(vec3 rd) { 
  return vec2(0.2 + 1 * atan(rd.z, rd.x) / (M_PI),0.5 + 1.0 * atan(rd.y, sqrt(rd.x * rd.x + rd.z * rd.z)) / (M_PI));
}

void main() { 
  if(isBack > 0) { 
    vec3 rd = pos.xyz;
    rd.z = 1;
    vec2 st = envUV(normalize(rd));
    st.y = 1 - st.y;
    color = texture2D(envMap, st).xyz;
  } else {

    vec3 eyeRd = normalize(pos.xyz - ro);
    vec3 reflectedRd = reflect(eyeRd, vec3(0,1,0));
    
    vec3 kd = texture2D(diffuseTexture, uv).xyz;
    vec3 ka = texture2D(envMap, envUV(reflectedRd)).xyz;
    float fresnel = 1 - abs(dot(reflectedRd, vec3(0,1,0)));
    fresnel = fresnel * fresnel * 0.6;
    color = (1 - fresnel) * kd +  ka * ka * fresnel;
  }
}
