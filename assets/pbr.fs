#version 330 core

uniform sampler2D u_envMap;
uniform sampler2D u_diffuseTexture;
uniform sampler2D u_specularTexture;
uniform sampler2D u_bumpTexture;
uniform vec3 u_kd;
uniform vec3 u_ka;
uniform vec3 u_ks;
uniform float u_shinnness;
uniform vec3 u_ro;
uniform vec3 u_rd;
uniform bool u_isBack;
uniform int  u_shadingMode;
uniform bool u_useTextures;

in vec4 f_pos;
in vec3 f_normal;
in vec2 f_uv;
out vec3 o_color;

#define M_PI 3.1415

vec2 envUV(vec3 rd) { 
  return vec2(0.2 + 1 * atan(rd.z, rd.x) / (M_PI),0.5 + 1.0 * atan(rd.y, sqrt(rd.x * rd.x + rd.z * rd.z)) / (M_PI));
}

vec3 getDiffuse(vec2 st) { 
  if(u_useTextures) { return texture2D(u_diffuseTexture, st).xyz; }
  return u_kd;
}
vec3 getSpecular(vec2 st) { 
  if(u_useTextures) { return texture2D(u_specularTexture, st).xyz; }
  return u_ks;
}
vec3 getAmbient(vec2 st) {
  return u_ka;
}

vec3 phongShading(vec3 I, vec3 L, vec3 N) { 
  vec3 kd = getDiffuse(f_uv);
  vec3 R = reflect(I, N);

  vec3 ka = texture2D(u_envMap, envUV(R)).xyz;

  float fresnel = 1 - abs(dot(R, N));
  fresnel = fresnel * fresnel * 0.6;
  vec3 color = (1 - fresnel) * kd +  ka * ka * fresnel;
  return vec3(N * 0.5 + 0.5);
}

vec3 backShading() { 
  vec3 rd = f_pos.xyz;
  rd.z = 1;
  vec2 st = envUV(normalize(rd));
  st.y = 1 - st.y;
  return texture2D(u_envMap, st).xyz;
}

void main() { 
  if(u_isBack) { 
    o_color = backShading();
  } else {
    vec3 eyeRd = normalize(f_pos.xyz - u_ro);
    if(u_shadingMode == 0) {
      o_color = phongShading(eyeRd, vec3(1,-1,1), f_normal);
    }
  }
}