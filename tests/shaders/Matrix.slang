#include "Matrix.h"

[shader("compute")]
[numthreads(32, 1, 1)]
void main() {
    const Matrix::Arguments arguments = rootArguments<Matrix::Arguments>();
    const MatrixA<half, 16, 16> a = MatrixA<half, 16, 16>.load(arguments.a, 16);
    const MatrixB<half, 16, 16> b = MatrixB<half, 16, 16>.load(arguments.b, 16);
    MatrixAccumulator<float, 16, 16> result = multiplyAdd(a, b, MatrixAccumulator<float, 16, 16>.load(arguments.bias, 16)) * 2.0;
    for (int index = 0; index < result.getCount(); ++index)
        result[index] = max(result[index], 0.0);
    result.store(arguments.result, 16);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void tiledMain(uint3 group : SV_GroupID) {
    const Matrix::TiledArguments arguments = rootArguments<Matrix::TiledArguments>();
    const uint row = group.y * 16;
    const uint column = group.x * 16;
    MatrixAccumulator<float, 16, 16> sum = MatrixAccumulator<float, 16, 16>.load(arguments.result + (row * arguments.size + column), arguments.size);
    for (uint k = 0; k < arguments.size; k += 16) {
        const MatrixA<half, 16, 16> a = MatrixA<half, 16, 16>.load(arguments.a + (row * arguments.size + k), arguments.size);
        const MatrixB<half, 16, 16> b = MatrixB<half, 16, 16>.load(arguments.b + (k * arguments.size + column), arguments.size);
        sum = multiplyAdd(a, b, sum);
    }
    sum.store(arguments.result + (row * arguments.size + column), arguments.size);
}
