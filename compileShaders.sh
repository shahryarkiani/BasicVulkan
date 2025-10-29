"$VULKAN_SDK/Bin/dxc" shaders/shader.frag.hlsl -T ps_6_0 -spirv -Fo shaders/frag.spv
"$VULKAN_SDK/Bin/dxc" shaders/shader.vert.hlsl -T vs_6_0 -spirv -Fo shaders/vert.spv
