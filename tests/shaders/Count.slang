#include "Count.h"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const Count::Arguments arguments = rootArguments<Count::Arguments>();
    if (id.x >= arguments.count)
        return;
    const uint value = arguments.values[id.x];
    atomicMax(arguments.counters + 1, value);
    if (value < arguments.threshold)
        return;
    arguments.slots[atomicAdd(arguments.counters, 1u)] = value;
}
