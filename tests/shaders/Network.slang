#include "Network.h"

struct Weights {
    Ptr<float> values;
    AtomicPtr<float> gradients;
};

[Differentiable]
[BackwardDerivative(loadBackward)]
float load(no_diff Weights weights, no_diff uint index) {
    return weights.values[index];
}

void loadBackward(Weights weights, uint index, float gradient) {
    atomicAdd(weights.gradients + index, gradient);
}

[Differentiable]
float evaluate(no_diff Weights weights, no_diff float4 input) {
    float output = load(weights, Network::parameterCount - 1);
    [ForceUnroll]
    for (uint row = 0; row < Network::hidden; ++row) {
        float sum = load(weights, Network::hidden * Network::inputs + row);
        [ForceUnroll]
        for (uint column = 0; column < Network::inputs; ++column)
            sum += load(weights, row * Network::inputs + column) * input[column];
        output += load(weights, Network::hidden * Network::inputs + Network::hidden + row) * max(sum, 0.0);
    }
    return output;
}

[Differentiable]
float getLoss(no_diff Weights weights, no_diff float4 input, no_diff float target) {
    const float error = evaluate(weights, input) - target;
    return error * error;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const Network::Arguments arguments = rootArguments<Network::Arguments>();
    if (id.x >= arguments.count)
        return;
    const Weights weights = {arguments.parameters, arguments.gradients};
    bwd_diff(getLoss)(weights, arguments.samples[id.x], arguments.targets[id.x], 1.0);
}
