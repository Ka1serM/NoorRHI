#include "Matrix.h"

[shader("compute")]
[numthreads(32, 1, 1)]
void main() {
    const Matrix::FloatArguments arguments = rootArguments<Matrix::FloatArguments>();
    const MatrixA<float, 16, 8> a = MatrixA<float, 16, 8>.load(arguments.a, 8);
    const MatrixB<float, 8, 16> b = MatrixB<float, 8, 16>.load(arguments.b, 16);
    multiplyAdd(a, b, MatrixAccumulator<float, 16, 16>.load(arguments.bias, 16)).store(arguments.result, 16);
}
