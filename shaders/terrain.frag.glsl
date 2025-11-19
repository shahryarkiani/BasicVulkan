#version 450

layout (location = 0) in VertexInput {
  vec4 color;
  vec3 position;
  vec3 normal;
} vertexInput;

layout(location = 0) out vec4 outFragColor;

vec3 lightPosition = vec3(10.0, 10.0, 10.0);

void main()
{
	vec3 toLight = normalize(lightPosition - vertexInput.position);
	float lightDot = dot(toLight, vertexInput.normal);
	outFragColor = vertexInput.color * (lightDot);
	outFragColor.w = 1.0;
}