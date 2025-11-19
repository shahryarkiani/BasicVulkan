#version 450
#extension GL_EXT_mesh_shader : require

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

const float horizontalOffset = 0.5;
const float verticalOffset = 0.5;
const float heightOffset = 1.0;

float getHeight(uint globalX, uint globalY) {
	float x = horizontalOffset * globalX;
	float y = verticalOffset * globalY;

	return sin(x / 2.0) + cos(y/ 4.0);
}

void main()
{
  SetMeshOutputsEXT(64, 98);
  uint x = gl_LocalInvocationID.x;
  uint y = gl_LocalInvocationID.y;
  uint idx = gl_LocalInvocationIndex;

  uint globalX = gl_WorkGroupID.x * 7 + x;
  uint globalY = gl_WorkGroupID.y * 7 + y;

  uint height = globalX * globalY;

  mat4 mvp = ubo.projection * ubo.view;
  // Create grid of vertices
  vec4 position = vec4(horizontalOffset * globalX, heightOffset * getHeight(globalX, globalY), verticalOffset * globalY, 1.0);
  gl_MeshVerticesEXT[idx].gl_Position = mvp * position;

  float hR = heightOffset * getHeight(globalX - 1, globalY);
  float hL = heightOffset * getHeight(globalX + 1, globalY);
  float hU = heightOffset * getHeight(globalX, globalY - 1);
  float hD = heightOffset * getHeight(globalX, globalY + 1);

  float heightDiffX = (hR - hL) / (2.0 * horizontalOffset);
  float heightDiffY = (hU - hD) / (2.0 * verticalOffset);

  vertexOutput[idx].color = colors[height % 3];
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