#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_ARB_shading_language_include : require

#include "psrdnoise2.glsl"

layout (binding = 0) uniform UBO
{
  mat4 model;
  mat4 view;
  mat4 projection;
} ubo;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 98) out;

layout(location = 0) out VertexOutput
{
  vec4 color;
  vec3 position;
  vec3 normal;
} vertexOutput[];

const vec4[3] colors = {
  vec4(0.0, 1.0, 0.0, 1.0),
  vec4(0.0, 0.0, 1.0, 1.0),
  vec4(1.0, 0.0, 0.0, 1.0)
};

vec4 red =   vec4(1.0, 0.0, 0.0, 0.0);
vec4 green = vec4(0.0, 1.0, 0.0, 0.0);
vec4 blue =  vec4(0.0, 0.0, 1.0, 0.0);

const float horizontalOffset = 0.05;
const float verticalOffset = 0.05;
const float heightOffset = 1.0;

float fbm(vec2 p, out vec2 gradientOut) {

    float G = 0.5;
    float f = 0.05;
    float a = 10.0;
    float t = 0.0;
    vec2 gradient = vec2(0.0, 0.0);
    for(int i = 0; i < 2; i++) {
        vec2 gradientChange;
        t += a * psrdnoise(f * p, vec2(0.0, 0.0), 0.0, gradientChange);
        gradient += a * gradientChange;
        f *= 2.0;
        a *= G;
    }
    gradientOut = gradient;
    return t;
}

float getHeight(uint globalX, uint globalY, out vec2 gradientOut) {
    vec2 p = vec2(globalX * horizontalOffset, globalY * verticalOffset);
    return fbm(p, gradientOut);
}

void main()
{
  SetMeshOutputsEXT(64, 98);
  uint x = gl_LocalInvocationID.x;
  uint y = gl_LocalInvocationID.y;
  uint idx = gl_LocalInvocationIndex;

  uint globalX = gl_WorkGroupID.x * 7 + x;
  uint globalY = gl_WorkGroupID.y * 7 + y;

  vec2 gradient;
  float height = getHeight(globalX, globalY, gradient);

  mat4 mvp = ubo.projection * ubo.view;
  // Create grid of vertices
  vec4 position = vec4(horizontalOffset * globalX, heightOffset * height, verticalOffset * globalY, 1.0);
  gl_MeshVerticesEXT[idx].gl_Position = mvp * position;

  float hR = heightOffset * getHeight(globalX - 1, globalY, gradient);
  float hL = heightOffset * getHeight(globalX + 1, globalY, gradient);
  float hU = heightOffset * getHeight(globalX, globalY - 1, gradient);
  float hD = heightOffset * getHeight(globalX, globalY + 1, gradient);

  float heightDiffX = (hR - hL) / (2.0 * horizontalOffset);
  float heightDiffY = (hU - hD) / (2.0 * verticalOffset);

  //vertexOutput[idx].color = vec4(0.15, 0.28, 0.38, 1.0);
  //if(height > -0.5) 
  vertexOutput[idx].color = vec4(0.05, 0.16, 0.08, 1.0);
  if(height > 1.7) vertexOutput[idx].color = (green + red + blue) * 0.9;

  vertexOutput[idx].normal = normalize(vec3(heightDiffX, 1.0, heightDiffY));
  vertexOutput[idx].position = (position).xyz;

  if (x == 7 || y == 7) return;
  // The threads that aren't at the border output two triangles each
  uint baseVertexIdx = idx;
  uint rightIdx = idx + 1;
  uint upIdx = idx + 8;
  uint upRightIdx = upIdx + 1;

  uint triIdx = (y * 7) + x;
  gl_PrimitiveTriangleIndicesEXT[triIdx * 2 + 0] =  uvec3(baseVertexIdx, rightIdx, upIdx);
  gl_PrimitiveTriangleIndicesEXT[triIdx * 2 + 1] =  uvec3(rightIdx, upRightIdx, upIdx);
}