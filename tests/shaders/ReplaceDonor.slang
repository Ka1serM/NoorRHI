#include "Replace.h"

[noinline]
uint offsetBy(uint value) {
    return value;
}

[noinline]
uint triple(uint value) {
    return value * 3u + uint(cos(0.0f)) - 1u;
}

[noinline]
uint shade(uint value) {
    return offsetBy(triple(value));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void donorMain(uint3 id : SV_DispatchThreadID) {
    const Replace::Arguments arguments = rootArguments<Replace::Arguments>();
    if (id.x < arguments.count)
        arguments.result[id.x] = shade(id.x);
}
