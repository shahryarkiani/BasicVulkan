struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR;
};

static const float2 positions[3] = {
    float2( 0.0f, -0.5f),
    float2( 0.5f,  0.5f),
    float2(-0.5f,  0.5f)
};

static const float3 colors[3] = {
    float3(1.0f, 0.0f, 0.0f), // Red
    float3(0.0f, 1.0f, 0.0f), // Green
    float3(0.0f, 0.0f, 1.0f)  // Blue
};

VSOutput main(uint vertexID : SV_VertexID)
{
    VSOutput output;
    output.position = float4(positions[vertexID], 0.0f, 1.0f);
    output.color = colors[vertexID];
    return output;
}