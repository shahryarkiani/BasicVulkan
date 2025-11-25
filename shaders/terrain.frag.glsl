#version 450

layout (location = 0) in VertexInput {
  vec4 color;
  vec3 position;
  vec3 normal;
} vertexInput;

layout (binding = 0) uniform UBO
{
  mat4 model;
  mat4 view;
  mat4 projection;
  vec3 forward;
  float hfov;
} ubo;

layout(location = 0) out vec4 outFragColor;

vec3 lightPosition = vec3(50.0, 250.0, 50.0);

void main()
{
	vec3 cameraPos = inverse(ubo.view)[3].xyz;
	vec3 toLight = normalize(lightPosition - vertexInput.position);
	float lightDot = dot(toLight, normalize(vertexInput.normal));
	lightDot = max(lightDot, 0.0);
	outFragColor = vertexInput.color * (lightDot + 0.2);
	outFragColor.w = 1.0;
}