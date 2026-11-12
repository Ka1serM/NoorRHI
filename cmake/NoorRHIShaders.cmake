#! enable_language works only at file scope, so the app includes this file in its top-level CMakeLists.txt.
include_guard(GLOBAL)

enable_language(ASM)

cmake_path(ABSOLUTE_PATH CMAKE_CURRENT_LIST_DIR NORMALIZE OUTPUT_VARIABLE noorrhi_cmake_dir)
cmake_path(GET noorrhi_cmake_dir PARENT_PATH noorrhi_root_dir)
set(NOORRHI_SOURCE_DIR "${noorrhi_root_dir}/src" CACHE INTERNAL "")
set(NOORRHI_SHADER_HEADER_SCRIPT "${noorrhi_cmake_dir}/NoorRHIShaderHeader.cmake" CACHE INTERNAL "")
find_program(NOORRHI_SLANGC slangc HINTS "$ENV{VULKAN_SDK}/bin" REQUIRED)

set(NOORRHI_FALLBACK_SCRIPT "${noorrhi_cmake_dir}/NoorRHIRayTracingFallback.cmake" CACHE INTERNAL "")

#! Compiles one Slang source to one module and adds it to the embedded data of the shader target.
function(noorrhi_compile_module source module reflection symbol)
    cmake_parse_arguments(PARSE_ARGV 4 compile "" "" "FLAGS;DEPENDS")
    get_filename_component(directory "${source}" DIRECTORY)
    get_filename_component(stem "${module}" NAME_WE)
    get_filename_component(output "${module}" DIRECTORY)
    set(shaders "${NOORRHI_SOURCE_DIR}/NoorRHI/Shaders")
    if(module MATCHES "[\"\\\\]")
        message(FATAL_ERROR "noorrhi_add_shaders: ${module} contains a quote or backslash")
    endif()
    get_filename_component(depfile "${module}" NAME)
    add_custom_command(OUTPUT "${module}" "${reflection}"
                       COMMAND "${CMAKE_COMMAND}" -E make_directory "${output}"
                       COMMAND "${NOORRHI_SLANGC}" "${source}" ${compile_FLAGS} -I "${directory}" -I "${shaders}" -I "${NOORRHI_SOURCE_DIR}"
                               -matrix-layout-row-major -o "${module}" -reflection-json "${reflection}" -depfile "${output}/${depfile}.d"
                       DEPENDS "${source}" ${compile_DEPENDS} "${shaders}/NoorRHI.slang" "${shaders}/NoorRHIWebGpu.slang" "${shaders}/NoorRHIRayTracing.slang"
                       DEPFILE "${output}/${depfile}.d"
                       VERBATIM)
    set(modules ${modules} "${module}" PARENT_SCOPE)
    if(WIN32)
        set(section ".section .rdata,\"dr\"")
    elseif(EMSCRIPTEN)
        set(section ".section .rodata.${symbol},\"\",@")
    else()
        set(section ".section .rodata.${symbol},\"a\"")
    endif()
    #! COFF has no .size directive.
    set(data_size "    .size ${symbol}, .L${symbol}_end - ${symbol}\n")
    set(size_size "    .size ${symbol}_size, 4\n")
    if(WIN32)
        set(data_size "")
        set(size_size "")
    endif()
    set(data "${data}\
    ${section}
    .globl ${symbol}
    .globl ${symbol}_size
    .p2align 4
${symbol}:
    .incbin \"${module}\"
.L${symbol}_end:
${data_size}    .p2align 2
${symbol}_size:
    .int .L${symbol}_end - ${symbol}
${size_size}" PARENT_SCOPE)
endfunction()

#! Compiles each Slang source, embeds the modules, and generates <name>.slang.h with a NoorRHI::Shader for each entry point. A source with ray tracing or cooperative matrices also gets modules for the fallbacks, and Shader::getCode chooses one when the pipeline opens.
function(noorrhi_compile_shaders target)
    #! The WebGPU backend translates SPIR-V 1.3 to WGSL when a pipeline opens, because the Tint SPIR-V reader reads at most SPIR-V 1.3.
    set(extension .spv)
    if(NOORRHI_BACKEND STREQUAL "WebGpu")
        set(flags -target spirv -profile spirv_1_3 -fvk-use-entrypoint-name -DNOORRHI_WEBGPU)
    else()
        set(flags -target spirv -profile spirv_1_5 -fvk-use-entrypoint-name -bindless-space-index 0 -Wno-41012)
    endif()
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${target}")
    set(headers "")
    set(modules "")
    set(data "")
    set(functions "")
    foreach(source IN LISTS ARGN)
        cmake_path(ABSOLUTE_PATH source NORMALIZE)
        cmake_path(GET source STEM name)
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
        file(READ "${source}" text)
        get_filename_component(source_directory "${source}" DIRECTORY)
        string(MAKE_C_IDENTIFIER "noorrhi_${target}_${name}" symbol)
        set(header "${output}/${name}.slang.h")
        set(header_arguments "-DHEADER=${header}" "-DNAME=${name}")
        set(header_depends "${NOORRHI_SHADER_HEADER_SCRIPT}")
        set(ray_tracing "")
        if(text MATCHES "\\[shader\\(\"(raygeneration|miss|closesthit|anyhit|intersection|callable)\"\\)\\]")
            set(ray_tracing pipeline)
        elseif(text MATCHES "traceRayInline")
            set(ray_tracing inline)
        endif()
        set(matrices OFF)
        if(text MATCHES "Matrix(A|B|Accumulator)[ \t]*<")
            set(matrices ON)
        endif()
        if(ray_tracing STREQUAL "pipeline")
            set(fallback "${output}/${name}.fallback.slang")
            set(entries "${output}/${name}.entries.cmake")
            add_custom_command(OUTPUT "${fallback}" "${entries}"
                               COMMAND "${CMAKE_COMMAND}" -E make_directory "${output}"
                               COMMAND "${CMAKE_COMMAND}" "-DSOURCE=${source}" "-DOUTPUT=${fallback}" "-DENTRIES=${entries}" -P "${NOORRHI_FALLBACK_SCRIPT}"
                               DEPENDS "${source}" "${NOORRHI_FALLBACK_SCRIPT}"
                               VERBATIM)
            list(APPEND header_arguments "-DENTRIES=${entries}")
            list(APPEND header_depends "${entries}")
        endif()
        #! Bit 0 of a variant is hardware ray tracing and bit 1 cooperative matrices. WebGPU has neither, and a source compiles only the bits that it uses.
        set(used 0)
        if(ray_tracing AND NOT NOORRHI_BACKEND STREQUAL "WebGpu")
            math(EXPR used "${used} | 1")
        endif()
        if(matrices AND NOT NOORRHI_BACKEND STREQUAL "WebGpu")
            math(EXPR used "${used} | 2")
        endif()
        set(codes "")
        foreach(index RANGE 3)
            math(EXPR variant "${index} & ${used}")
            list(APPEND codes "${symbol}_${variant}")
            if(index GREATER variant)
                continue()
            endif()
            set(variant_source "${source}")
            set(variant_flags ${flags})
            set(variant_depends "")
            math(EXPR native_ray_tracing "${variant} & 1")
            math(EXPR native_matrices "${variant} & 2")
            if(ray_tracing AND NOT native_ray_tracing)
                list(APPEND variant_flags -DNOORRHI_RAYTRACING_FALLBACK)
            endif()
            if(ray_tracing STREQUAL "pipeline" AND NOT native_ray_tracing)
                set(variant_source "${fallback}")
                list(APPEND variant_flags -DNOORRHI_RAYTRACING_LAUNCH -I "${source_directory}")
                set(variant_depends "${entries}")
            endif()
            if(matrices AND NOT native_matrices)
                list(APPEND variant_flags -DNOORRHI_MATRIX_FALLBACK)
            endif()
            noorrhi_compile_module("${variant_source}" "${output}/${name}.${variant}${extension}" "${output}/${name}.${variant}.json" "${symbol}_${variant}"
                                   FLAGS ${variant_flags} DEPENDS ${variant_depends})
            list(APPEND header_depends "${output}/${name}.${variant}${extension}")
        endforeach()
        if(NOT ray_tracing STREQUAL "pipeline")
            list(APPEND header_arguments "-DREFLECTION=${output}/${name}.0.json")
            list(APPEND header_depends "${output}/${name}.0.json")
        endif()
        string(REPLACE ";" "," codes "${codes}")
        list(APPEND header_arguments "-DCODES=${codes}")
        add_custom_command(OUTPUT "${header}"
                           COMMAND "${CMAKE_COMMAND}" ${header_arguments} -P "${NOORRHI_SHADER_HEADER_SCRIPT}"
                           DEPENDS ${header_depends}
                           VERBATIM)
        list(APPEND headers "${header}")
    endforeach()

    set(data_source "${output}/${target}.S")
    if(NOT EMSCRIPTEN AND NOT WIN32)
        string(APPEND data "    .section .note.GNU-stack,\"\",@progbits\n")
    endif()
    file(CONFIGURE OUTPUT "${data_source}" CONTENT "${data}" @ONLY)

    add_library(${target} OBJECT "${data_source}" ${headers})
    set_source_files_properties("${data_source}" PROPERTIES OBJECT_DEPENDS "${modules}")
    target_include_directories(${target} PUBLIC "${output}")
endfunction()

function(noorrhi_add_shaders target)
    noorrhi_compile_shaders(${target} ${ARGN})
    target_link_libraries(${target} PUBLIC NoorRHI::NoorRHI)
endfunction()
