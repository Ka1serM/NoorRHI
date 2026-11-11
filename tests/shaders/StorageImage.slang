#include "StorageImage.h"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const StorageImage::Arguments arguments = rootArguments<StorageImage::Arguments>();
    if (id.x >= arguments.size || id.y >= arguments.size)
        return;
    const float blue = arguments.source.SampleLevel(arguments.sampler, float2(0.5, 0.5), 0.0).b;
    arguments.target[id.xy] = float4(float(id.x * 16u) / 255.0, float(id.y * 16u) / 255.0, blue, 1.0);
}
