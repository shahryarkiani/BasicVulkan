struct PSInput {
    float3 color : COLOR;
    float2 inTexCoord: TEXCOORD0;
};

Texture2D texture : register(t1);
SamplerState texSampler : register(s1);

float4 main(PSInput input) : SV_Target
{
    return texture.Sample(texSampler, input.inTexCoord);
}