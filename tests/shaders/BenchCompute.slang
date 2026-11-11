#include "Bench.h"

[shader("compute")]
[numthreads(256, 1, 1)]
void streamMain(uint3 id : SV_DispatchThreadID) {
    const Bench::StreamArguments arguments = rootArguments<Bench::StreamArguments>();
    if (id.x < arguments.count)
        arguments.destination[id.x] = arguments.source[id.x] * 2.0f + 1.0f;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void gemmMain(uint3 group : SV_GroupID) {
    const Bench::GemmArguments arguments = rootArguments<Bench::GemmArguments>();
    const uint size = arguments.size;
    const uint row = group.y * 16;
    const uint column = group.x * 16;
    MatrixAccumulator<float, 16, 16> sum = MatrixAccumulator<float, 16, 16>.load(arguments.c + (row * size + column), size);
    for (uint k = 0; k < size; k += 16) {
        const MatrixA<half, 16, 16> a = MatrixA<half, 16, 16>.load(arguments.a + (row * size + k), size);
        const MatrixB<half, 16, 16> b = MatrixB<half, 16, 16>.load(arguments.b + (k * size + column), size);
        sum = multiplyAdd(a, b, sum);
    }
    sum.store(arguments.c + (row * size + column), size);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void scalarGemmMain(uint3 id : SV_DispatchThreadID) {
    const Bench::GemmArguments arguments = rootArguments<Bench::GemmArguments>();
    const uint size = arguments.size;
    float sum = arguments.c[id.y * size + id.x];
    for (uint k = 0; k < size; ++k)
        sum += float(arguments.a[id.y * size + k]) * float(arguments.b[k * size + id.x]);
    arguments.c[id.y * size + id.x] = sum;
}
