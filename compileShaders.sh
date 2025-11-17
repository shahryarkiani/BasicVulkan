"$VULKAN_SDK/Bin/dxc" shaders/shader.frag.hlsl -T ps_6_0 -spirv -Fo shaders/frag.spv
"$VULKAN_SDK/Bin/dxc" shaders/shader.vert.hlsl -T vs_6_0 -spirv -Fo shaders/vert.spv
glslangValidator --target-env vulkan1.3 -V -S frag shaders/terrain.frag.glsl -o shaders/terrain.frag.spv
glslangValidator --target-env vulkan1.3 -V -S mesh shaders/terrain.mesh.glsl -o shaders/terrain.mesh.spv
glslangValidator --target-env vulkan1.3 -V -S task shaders/terrain.task.glsl -o shaders/terrain.task.spv
