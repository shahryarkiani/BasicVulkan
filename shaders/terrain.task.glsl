#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_ARB_shading_language_include : require

#include "mesh_shader_shared.h"

taskPayloadSharedEXT MeshPayload meshPayload;

layout(local_size_x = 29, local_size_y = 29, local_size_z = 1) in;

layout (binding = 0) uniform UBO
{
  mat4 model;
  mat4 view;
  mat4 projection;
} ubo;

void main()
{
    int x = int(gl_WorkGroupID.x) - 14;
    int y = int(gl_WorkGroupID.y) - 14;

    // Each grid cube will have a size of 179.2 = baseChunkCount * baseGridOffset * 7
    int baseGridCount = 256;
    float baseGridOffset = 0.1;
    float gridSize = baseGridCount * baseGridOffset * 7;

    // TODO: Probably don't even need to call inverse
    vec2 cameraPos = inverse(ubo.view)[3].xz;
    // We want the camera to be at the center of the grid for now, this will be changed
    // when we start doing culling
    cameraPos -= vec2(gridSize / 2);

    // We want to snap the camera pos to the nearest grid
    vec2 snappedPos = floor(cameraPos / gridSize) * gridSize; 

    vec3 basePos = vec3(gridSize * x + snappedPos.x, 0.0, gridSize * y + snappedPos.y);

	// TODO: Frustum Culling
    int dist = max(abs(x), abs(y));
    // stretch the LODs
    if(dist != 0) dist = (dist / 2) + 1;
    int modifier = int(pow(2.0, dist));
    meshPayload.basePosition = basePos;
    meshPayload.gridOffset = baseGridOffset * modifier;
    EmitMeshTasksEXT(baseGridCount / modifier, baseGridCount / modifier,1);
}