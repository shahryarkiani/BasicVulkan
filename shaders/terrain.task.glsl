#version 450
#extension GL_EXT_mesh_shader : require
#extension GL_ARB_shading_language_include : require

#include "mesh_shader_shared.h"

taskPayloadSharedEXT MeshPayload meshPayload;

void main()
{
	// TODO: Frustum Culling
	meshPayload.basePosition = vec3(0.0, 0.0, 0.0);
	meshPayload.gridOffset = 0.1;
	EmitMeshTasksEXT(500, 500, 1);
}