# The NoorRHI test suite. Catch2 and slangc are test-only dependencies; the
# library itself needs neither. SDL3 is optional and gates the presentation
# test, so the suite still builds on a machine without it.
find_program(NOORRHI_SLANGC_EXECUTABLE NAMES slangc
    HINTS ${NOORRHI_VULKAN_SDK_BIN_HINTS}
    REQUIRED)

if(NOT TARGET Catch2::Catch2WithMain)
    find_package(Catch2 3 QUIET)
endif()
if(NOT TARGET Catch2::Catch2WithMain)
    # An explicit source directory, then a vendored copy inside this repository.
    set(NOORRHI_CATCH2_DIR "" CACHE PATH "Catch2 v3 source directory")
    foreach(_catch2
            "${NOORRHI_CATCH2_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/../external/Catch2")
        if(_catch2 AND EXISTS "${_catch2}/CMakeLists.txt")
            add_subdirectory("${_catch2}" "${CMAKE_BINARY_DIR}/external/Catch2")
            break()
        endif()
    endforeach()
endif()
if(NOT TARGET Catch2::Catch2WithMain)
    message(FATAL_ERROR
        "Catch2 v3 not found. Install it, set NOORRHI_CATCH2_DIR, or configure "
        "with -DGPU_BUILD_TESTS=OFF.")
endif()
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch OPTIONAL RESULT_VARIABLE NOORRHI_CATCH_MODULE)

set(NOORRHI_TEST_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(NOORRHI_ADD_SHADER "${NOORRHI_TEST_SHADER_DIR}/add.spv")
set(NOORRHI_TRIANGLE_SHADER "${NOORRHI_TEST_SHADER_DIR}/triangle.spv")
set(NOORRHI_RAYTRACING_SHADER "${NOORRHI_TEST_SHADER_DIR}/raytracing.spv")
set(NOORRHI_IMAGE_SHADER "${NOORRHI_TEST_SHADER_DIR}/image.spv")

# Most tests use BDA buffers and framebuffer attachments, not shader textures,
# so only image.slang requests the descriptor-heap capability.
set(NOORRHI_TEST_SHADER_FLAGS
    -emit-spirv-directly
    -matrix-layout-row-major -fvk-use-entrypoint-name -Wno-30081)

add_custom_command(
    OUTPUT "${NOORRHI_ADD_SHADER}" "${NOORRHI_TRIANGLE_SHADER}" "${NOORRHI_RAYTRACING_SHADER}"
        "${NOORRHI_IMAGE_SHADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${NOORRHI_TEST_SHADER_DIR}"
    COMMAND "${NOORRHI_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/image.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5 -capability spvDescriptorHeapEXT ${NOORRHI_TEST_SHADER_FLAGS}
        -o "${NOORRHI_IMAGE_SHADER}"
    COMMAND "${NOORRHI_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/add.slang"
        -entry main -stage compute -target spirv
        -profile spirv_1_5 ${NOORRHI_TEST_SHADER_FLAGS}
        -o "${NOORRHI_ADD_SHADER}"
    COMMAND "${NOORRHI_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        -entry vertMain -stage vertex -entry fragMain -stage fragment
        -target spirv -profile spirv_1_5 ${NOORRHI_TEST_SHADER_FLAGS}
        -o "${NOORRHI_TRIANGLE_SHADER}"
    COMMAND "${NOORRHI_SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytracing.slang"
        -entry rayGenMain -stage raygeneration
        -entry missMain -stage miss
        -entry closestHitMain -stage closesthit
        -target spirv -profile spirv_1_5+spvRayTracingKHR ${NOORRHI_TEST_SHADER_FLAGS}
        -o "${NOORRHI_RAYTRACING_SHADER}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/add.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/triangle.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/raytracing.slang"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/image.slang"
    COMMENT "Compiling NoorRHI test Slang shaders" VERBATIM)

add_custom_target(noorrhi_test_shaders DEPENDS
    "${NOORRHI_ADD_SHADER}" "${NOORRHI_TRIANGLE_SHADER}" "${NOORRHI_RAYTRACING_SHADER}"
    "${NOORRHI_IMAGE_SHADER}")

function(add_noorrhi_test target source)
    add_executable(${target} "${source}")
    add_dependencies(${target} noorrhi_test_shaders)
    target_compile_features(${target} PRIVATE cxx_std_23)
    target_link_libraries(${target} PRIVATE NoorRHI::NoorRHI Catch2::Catch2WithMain)
    target_compile_definitions(${target} PRIVATE ${ARGN})
    if(COMMAND catch_discover_tests)
        catch_discover_tests(${target} TEST_PREFIX "NoorRHI:"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" PROPERTIES LABELS "NoorRHI")
    else()
        add_test(NAME "NoorRHI:${target}" COMMAND ${target})
    endif()
endfunction()

add_noorrhi_test(noorrhi_compute_test ComputeTest.cpp
    "NOORRHI_TEST_SHADER=\"${NOORRHI_ADD_SHADER}\""
    "NOORRHI_TRIANGLE_SHADER=\"${NOORRHI_TRIANGLE_SHADER}\""
    "NOORRHI_IMAGE_SHADER=\"${NOORRHI_IMAGE_SHADER}\"")

add_noorrhi_test(noorrhi_ray_tracing_test RayTracingTest.cpp
    "NOORRHI_RAYTRACING_SHADER=\"${NOORRHI_RAYTRACING_SHADER}\"")

# Presentation needs a real window system. Skipped entirely when no SDL3 is
# available, rather than forcing a windowing dependency on the whole suite.
if(NOT TARGET SDL3::SDL3-static AND NOT TARGET SDL3::SDL3)
    find_package(SDL3 QUIET)
endif()
if(TARGET SDL3::SDL3-static OR TARGET SDL3::SDL3)
    add_noorrhi_test(noorrhi_presentation_test PresentationTest.cpp)
    if(TARGET SDL3::SDL3-static)
        target_link_libraries(noorrhi_presentation_test PRIVATE SDL3::SDL3-static)
    else()
        target_link_libraries(noorrhi_presentation_test PRIVATE SDL3::SDL3)
    endif()
else()
    message(STATUS "NoorRHI: SDL3 not found, skipping noorrhi_presentation_test")
endif()
