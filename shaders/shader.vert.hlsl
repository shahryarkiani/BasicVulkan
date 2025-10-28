cbuffer UniformBufferObject : register(b0)
{
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR;
};

struct VSInput {
    float2 inPosition : POSITION; // layout(location = 0)
    float3 inColor    : COLOR;    // layout(location = 1)
};

VSOutput main(VSInput input)
{
    VSOutput output;
    float4x4 transform = mul(mul(proj, view), model);
    output.position = mul(transform, float4(input.inPosition, 0.0, 1.0));
    output.color = input.inColor;
    return output;
}
