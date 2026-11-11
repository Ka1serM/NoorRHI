#include "Add.h"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    const Add::Arguments arguments = rootArguments<Add::Arguments>();
    if (id.x < arguments.count)
        arguments.result[id.x] = arguments.a[id.x] + arguments.b[id.x];
}
