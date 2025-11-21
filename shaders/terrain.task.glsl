#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_ARB_shading_language_include : require

#include "mesh_shader_shared.h"

taskPayloadSharedEXT MeshPayload meshPayload;

layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

void main()
{
	// TODO: Frustum Culling
    if (gl_WorkGroupID.x == 0) {
        meshPayload.basePosition = vec3(0, 0, 0);
        meshPayload.gridOffset = 0.1;
        EmitMeshTasksEXT(256,256,1);
    }
    else if (gl_WorkGroupID.x == 1) {
        meshPayload.basePosition = vec3(0, 0, 179.2);
        meshPayload.gridOffset = 0.8;
        EmitMeshTasksEXT(32,32,1);
    }
}