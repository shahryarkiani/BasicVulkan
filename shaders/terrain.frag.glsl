#version 450

layout (location = 0) in VertexInput {
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
  float time;
} ubo;

layout(location = 0) out vec4 outFragColor;

const vec3 LIGHT_POS = vec3(50.0, 500.0, 50.0);
const vec3 COLOR_SKY = vec3(0.5, 0.6, 0.7);
const vec3 COLOR_GRASS = vec3(0.1, 0.25, 0.1);
const vec3 COLOR_ROCK  = vec3(0.35, 0.3, 0.3);
const vec3 COLOR_SNOW  = vec3(0.95, 0.95, 1.0);

void main()
{
    vec3 N = vertexInput.normal;
    vec3 camPos = inverse(ubo.view)[3].xyz;
    vec3 V = normalize(camPos - vertexInput.position);
    vec3 L = normalize(LIGHT_POS - vertexInput.position);
    
    float height = vertexInput.position.y;
    
    float slope = dot(N, vec3(0.0, 1.0, 0.0));
    
    vec3 terrainColor = COLOR_GRASS;
    
    float rockMix = smoothstep(10.0, 25.0, height);
    float snowMix = smoothstep(45.0, 60.0, height);
    
    float steepness = 1.0 - smoothstep(0.4, 0.7, slope);
    
    terrainColor = mix(terrainColor, COLOR_ROCK, rockMix);
    terrainColor = mix(terrainColor, COLOR_ROCK, steepness);
    
    // snow is based on height and steepness
    float snowFactor = snowMix * (1.0 - steepness * 0.8);
    terrainColor = mix(terrainColor, COLOR_SNOW, snowFactor);

    // diffuse
    float diff = max(dot(N, L), 0.0);
    
    // specular
    vec3 H = normalize(L + V);
    float specStrength = (snowFactor > 0.5) ? 0.8 : 0.1; 
    float shininess = (snowFactor > 0.5) ? 32.0 : 8.0;
    float spec = pow(max(dot(N, H), 0.0), shininess) * specStrength;
    
    // ambient
    vec3 ambient = 0.15 * terrainColor;
    
    vec3 finalColor = ambient + diff * terrainColor + vec3(spec);

    // Fog for distant terrain
    float distanceToCam = length(vertexInput.position - camPos);
    float fogDensity = 0.00006;
    float fogFactor = 1.0 - exp(-distanceToCam * fogDensity);
    
    finalColor = mix(finalColor, COLOR_SKY, fogFactor);

    outFragColor = vec4(finalColor, 1.0);
}