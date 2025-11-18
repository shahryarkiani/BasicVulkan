#version 450
#extension GL_EXT_mesh_shader : require

layout (binding = 0) uniform UBO
{
  mat4 model;
  mat4 view;
  mat4 projection;
} ubo;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(triangles, max_vertices = 64, max_primitives = 64) out;

layout(location = 0) out VertexOutput
{
  vec4 color;
} vertexOutput[];

const vec4[3] colors = {
  vec4(0.0, 1.0, 0.0, 1.0),
  vec4(0.0, 0.0, 1.0, 1.0),
  vec4(1.0, 0.0, 0.0, 1.0)
};

void main()
{
  uint x = gl_LocalInvocationID.x;
  uint y = gl_LocalInvocationID.y;
  uint idx = gl_LocalInvocationIndex;


  vec4 basePos = vec4(0.0, 0.0, 0.0, 1.0);
  vec4 horizontalOffset = vec4(1.0, 0.0, 0.0, 0.0);
  vec4 verticalOffset = vec4(0.0, 0.0, 1.0, 0.0);
  vec4 heightOffset = vec4(0.0, 1.0, 1.0, 0.0);
  
  SetMeshOutputsEXT(64, 98);

  mat4 mvp = ubo.projection * ubo.view;
  // Create grid of vertices
  gl_MeshVerticesEXT[idx].gl_Position = mvp * (basePos + horizontalOffset * x + verticalOffset * y);
  vertexOutput[idx].color = colors[idx % 3];
  
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