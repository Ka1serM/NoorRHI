# The examples are the NoorRHI library's own demonstration programs. They need
# slangc to produce SPIR-V; the library itself does not.
find_program(NOORRHI_SLANGC_EXECUTABLE NAMES slangc
    HINTS ${NOORRHI_VULKAN_SDK_BIN_HINTS}
    REQUIRED)

set(NOORRHI_EXAMPLE_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(NOORRHI_EXAMPLE_COMPUTE_SHADER "${NOORRHI_EXAMPLE_SHADER_DIR}/compute.spv")
set(NOORRHI_EXAMPLE_GRAPHICS_SHADER "${NOORRHI_EXAMPLE_SHADER_DIR}/triangle.spv")
set(NOORRHI_EXAMPLE_RAYGEN_SHADER "${NOORRHI_EXAMPLE_SHADER_DIR}/raygen.spv")

# Compiled for the descriptor-heap model the library's texture handles use.
# The resource heap stride stays at Slang's default (the image descriptor
# size), which is how the device lays out its slots.
set(NOORRHI_EXAMPLE_SHADER_FLAGS
    -capability spvDescriptorHeapEXT -emit-spirv-directly
    -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081)

add_custom_command(
    OUTPUT "${NOORRHI_EXAMPLE_COMPUTE_SHADER}"
           "${NOORRHI_EXAMPLE_GRAPHICS_SHADER}"
           "${NOORRHI_EXAMPLE_RAYGEN_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NOORRHI_EXAMPLE_SHADER_DIR}"
    COMMAND "${NOORRHI_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5+SPV_EXT_descriptor_heap
        ${NOORRHI_EXAMPLE_SHADER_FLAGS}
        -o "${NOORRHI_EXAMPLE_COMPUTE_SHADER}"
    COMMAND "${NOORRHI_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5+SPV_EXT_descriptor_heap
        ${NOORRHI_EXAMPLE_SHADER_FLAGS}
        -o "${NOORRHI_EXAMPLE_GRAPHICS_SHADER}"
    COMMAND "${NOORRHI_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygen.slang"
        -entry main -stage raygeneration -target spirv
        -profile spirv_1_5+spvRayTracingKHR+SPV_EXT_descriptor_heap
        ${NOORRHI_EXAMPLE_SHADER_FLAGS}
        -o "${NOORRHI_EXAMPLE_RAYGEN_SHADER}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raygen.slang"
    COMMENT "Compiling NoorRHI API example Slang shaders"
    VERBATIM)

add_custom_target(noorrhi_example_shaders DEPENDS
    "${NOORRHI_EXAMPLE_COMPUTE_SHADER}"
    "${NOORRHI_EXAMPLE_GRAPHICS_SHADER}"
    "${NOORRHI_EXAMPLE_RAYGEN_SHADER}")

function(add_noorrhi_example target source)
    add_executable(${target} "${source}")
    add_dependencies(${target} noorrhi_example_shaders)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_link_libraries(${target} PRIVATE NoorRHI::NoorRHI)
    target_compile_definitions(${target} PRIVATE ${ARGN})
endfunction()

add_noorrhi_example(noorrhi_compute_example compute.cpp
    "NOORRHI_COMPUTE_SHADER=\"${NOORRHI_EXAMPLE_COMPUTE_SHADER}\"")
add_noorrhi_example(noorrhi_graphics_example graphics.cpp
    "NOORRHI_GRAPHICS_SHADER=\"${NOORRHI_EXAMPLE_GRAPHICS_SHADER}\"")
add_noorrhi_example(noorrhi_raytracing_example raytracing.cpp
    "NOORRHI_RAYGEN_SHADER=\"${NOORRHI_EXAMPLE_RAYGEN_SHADER}\"")
