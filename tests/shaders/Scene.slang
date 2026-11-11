#include "Scene.h"

struct Varyings {
    float4 position : SV_Position;
    nointerpolation uint material : MATERIAL;
};

//! The Vulkan IDs include the first vertex, the vertex offset and the first instance on SPIR-V and on WGSL alike.
[shader("vertex")]
Varyings vertMain(uint vertex : SV_VulkanVertexID, uint instance : SV_VulkanInstanceID) {
    const Scene::DrawArguments arguments = rootArguments<Scene::DrawArguments>();
    const Scene::Instance item = arguments.instances[instance];
    Varyings output;
    //! The cells count rows from the top, and clip space Y points up.
    const float2 position = (float2(item.cell) + arguments.corners[vertex]) * arguments.cellSize - 1.0;
    output.position = float4(position.x, -position.y, 0.0, 1.0);
    output.material = item.material;
    return output;
}

[shader("fragment")]
float4 fragMain(Varyings input) : SV_Target0 {
    const Scene::DrawArguments arguments = rootArguments<Scene::DrawArguments>();
    return arguments.materials[input.material].texture.SampleLevel(arguments.sampler, float2(0.5, 0.5), 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 id : SV_DispatchThreadID) {
    const Scene::CullArguments arguments = rootArguments<Scene::CullArguments>();
    if (id.x >= arguments.instanceCount || arguments.instances[id.x].visible == 0)
        return;
    const uint slot = atomicAdd(arguments.count, 1u);
    if (arguments.indexed != 0) {
        DrawIndexedArgs draw;
        draw.indexCount = 6;
        draw.instanceCount = 1;
        draw.firstIndex = 6;
        draw.vertexOffset = 4;
        draw.firstInstance = id.x;
        arguments.indexedDraws[slot] = draw;
        return;
    }
    DrawArgs draw;
    draw.vertexCount = 6;
    draw.instanceCount = 1;
    draw.firstVertex = 8;
    draw.firstInstance = id.x;
    arguments.draws[slot] = draw;
}
