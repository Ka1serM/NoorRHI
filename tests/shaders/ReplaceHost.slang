#include "Replace.h"

[noinline]
uint offsetBy(uint value) {
    return value + 1000u;
}

[noinline]
uint shade(uint value) {
    return offsetBy(value) - 1000u - value;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void hostMain(uint3 id : SV_DispatchThreadID) {
    const Replace::Arguments arguments = rootArguments<Replace::Arguments>();
    if (id.x < arguments.count)
        arguments.result[id.x] = shade(id.x);
}
