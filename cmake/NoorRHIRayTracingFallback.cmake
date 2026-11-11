#! Runs with cmake -P. Writes OUTPUT, a copy of SOURCE where each ray generation shader becomes a compute kernel and traceRay calls the miss and hit shaders directly, and ENTRIES, a CMake file that lists the ray tracing entry points of each stage in source order.
set(stages raygeneration miss closesthit anyhit intersection callable)
file(READ "${SOURCE}" text)
string(REGEX MATCHALL "\\[shader\\(\"(raygeneration|miss|closesthit|anyhit|intersection|callable)\"\\)\\][^(]*\\([^,)]*" declarations "${text}")
foreach(stage IN LISTS stages)
    set(entries_${stage} "")
endforeach()
set(payloads "")
foreach(declaration IN LISTS declarations)
    string(REGEX MATCH "\"([a-z]+)\"" ignored "${declaration}")
    set(stage "${CMAKE_MATCH_1}")
    string(REGEX REPLACE "^\\[shader\\(\"[a-z]+\"\\)\\]" "" signature "${declaration}")
    string(REGEX MATCH "([A-Za-z_][A-Za-z0-9_]*)[ \t\r\n]*\\(" ignored "${signature}")
    set(name "${CMAKE_MATCH_1}")
    list(APPEND entries_${stage} "${name}")
    if(stage MATCHES "^(miss|closesthit|anyhit)$")
        string(REGEX MATCH "inout[ \t\r\n]+([A-Za-z_][A-Za-z0-9_:]*)" ignored "${declaration}")
        if(NOT CMAKE_MATCH_1)
            message(FATAL_ERROR "${SOURCE}: the ${stage} shader ${name} needs an inout payload as its first parameter")
        endif()
        string(MAKE_C_IDENTIFIER "${CMAKE_MATCH_1}" payload)
        set(type_${payload} "${CMAKE_MATCH_1}")
        list(APPEND payloads "${payload}")
        list(LENGTH entries_${stage} count)
        math(EXPR index "${count} - 1")
        list(APPEND ${stage}_${payload} "${index}:${name}")
    endif()
endforeach()
list(REMOVE_DUPLICATES payloads)

string(REGEX REPLACE "\\[shader\\(\"(raygeneration|miss|closesthit|anyhit|intersection|callable)\"\\)\\]" "" text "${text}")

function(write_switch output stage payload call)
    set(cases "")
    foreach(item IN LISTS ${stage}_${payload})
        string(REGEX MATCH "^([0-9]+):(.*)$" ignored "${item}")
        string(APPEND cases "        case ${CMAKE_MATCH_1}: ${CMAKE_MATCH_2}(${call}); break;\n")
    endforeach()
    set(${output} "switch (shader) {\n${cases}        default: break;\n        }" PARENT_SCOPE)
endfunction()

set(generated "")
foreach(payload IN LISTS payloads)
    write_switch(miss miss ${payload} "this")
    write_switch(closest closesthit ${payload} "this, attributes")
    write_switch(any anyhit ${payload} "this, attributes")
    string(APPEND generated "
extension ${type_${payload}} : IRayPayload {
    [mutating] void noorrhiMiss(uint shader) {
        ${miss}
    }
    [mutating] void noorrhiClosestHit(uint shader, BuiltInTriangleIntersectionAttributes attributes) {
        ${closest}
    }
    [mutating] void noorrhiAnyHit(uint shader, BuiltInTriangleIntersectionAttributes attributes) {
        ${any}
    }
}
")
endforeach()
foreach(name IN LISTS entries_raygeneration)
    string(APPEND generated "
[shader(\"compute\")]
[numthreads(8, 8, 1)]
void noorrhi_raygen_${name}(uint3 id : SV_DispatchThreadID) {
    if (beginRayLaunch(id))
        ${name}();
}
")
endforeach()

file(WRITE "${OUTPUT}" "#line 1 \"${SOURCE}\"\n${text}\n#line 1 \"${OUTPUT}\"\n${generated}")
set(list_file "")
foreach(stage IN LISTS stages)
    string(APPEND list_file "set(fallback_${stage} \"${entries_${stage}}\")\n")
endforeach()
file(WRITE "${ENTRIES}" "${list_file}")
