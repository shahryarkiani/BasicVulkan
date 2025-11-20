#version 450
#extension GL_EXT_mesh_shader : require

void main()
{
	// TODO: Frustum Culling
	EmitMeshTasksEXT(1000, 1000, 1);
}