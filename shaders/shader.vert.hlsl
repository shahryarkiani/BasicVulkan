cbuffer UniformBufferObject : register(b0)
{
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR;
    float2 outTexCoord : TEXCOORD0;
};

struct VSInput {
    float3 inPosition : POSITION;  // layout(location = 0)
    float3 inColor    : COLOR;     // layout(location = 1)
    float2 inTexCoord : TEXCOORD0; // layout(location = 2)
};

VSOutput main(VSInput input)
{
    VSOutput output;
    float4x4 transform = mul(proj, view);
    input.inPosition += float3(50.0, 15.0, 50.0);
    output.position = mul(transform, float4(input.inPosition, 1.0));
    output.color = input.inColor;
    output.outTexCoord = input.inTexCoord;
    return output;
}
