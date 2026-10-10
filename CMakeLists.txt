# The NoorRHI library is self-contained: Vulkan, Vulkan-HPP and VMA are its only
# dependencies. It knows nothing about NoorRay, shader compilers or assets.
#
# This file works two ways: as add_subdirectory() from a host project, and as
# the top-level CMakeLists of a standalone build. Everything that only makes
# sense for the standalone case is guarded by NOORRHI_IS_TOP_LEVEL.
cmake_minimum_required(VERSION 3.25)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NoorRHI VERSION 0.1.0 LANGUAGES CXX)
    set(NOORRHI_IS_TOP_LEVEL ON)
    set(NOORRHI_VERSION "${PROJECT_VERSION}")
else()
    set(NOORRHI_IS_TOP_LEVEL OFF)
    # Not a project() here, so PROJECT_VERSION belongs to the host. Keep our own.
    set(NOORRHI_VERSION "0.1.0")
endif()

option(NOORRHI_BUILD_EXAMPLES "Build the NoorRHI API examples" ${NOORRHI_IS_TOP_LEVEL})
option(NOORRHI_BUILD_TESTS "Build the NoorRHI API tests" ${NOORRHI_IS_TOP_LEVEL})

if(NOORRHI_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(GNUInstallDirs)

find_package(Vulkan REQUIRED)

# VMA ships only headers plus one implementation TU, which is ours to compile
# (src/vma_impl.cpp). Prefer an installed package; fall back to the vendored
# copy so a fresh clone with submodules configures without a system VMA.
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    find_package(VulkanMemoryAllocator QUIET)
endif()
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    # An explicit source directory, then a vendored copy inside this repository.
    # A host project that adds NoorRHI as a subdirectory usually already defines
    # the target, which the check above picks up.
    set(NOORRHI_VMA_DIR "" CACHE PATH "VulkanMemoryAllocator source directory")
    foreach(_candidate
            "${NOORRHI_VMA_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/external/VulkanMemoryAllocator")
        if(_candidate AND EXISTS "${_candidate}/CMakeLists.txt")
            add_subdirectory("${_candidate}"
                "${CMAKE_CURRENT_BINARY_DIR}/VulkanMemoryAllocator" EXCLUDE_FROM_ALL)
            break()
        endif()
    endforeach()
endif()
if(NOT TARGET GPUOpen::VulkanMemoryAllocator)
    message(FATAL_ERROR
        "VulkanMemoryAllocator not found. Install it, set NOORRHI_VMA_DIR, or "
        "populate ${CMAKE_CURRENT_SOURCE_DIR}/external/VulkanMemoryAllocator.")
endif()

add_library(NoorRHI STATIC
    src/device.cpp
    src/graphics.cpp
    src/swapchain.cpp
    src/image.cpp
    src/vma_impl.cpp
    src/pipeline_cache_file.cpp
    src/internal.hpp)
add_library(NoorRHI::NoorRHI ALIAS NoorRHI)

set_target_properties(NoorRHI PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(NoorRHI PUBLIC cxx_std_23)

# Only the public headers are visible to consumers; src/ holds internal.hpp,
# which pulls in Vulkan-HPP and VMA and must not leak into dependents.
target_include_directories(NoorRHI
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

target_compile_definitions(NoorRHI PUBLIC VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
# VMA is a private implementation detail: it appears in src/internal.hpp and
# src/vma_impl.cpp only, never in include/noorrhi. It is header-only, so nothing of
# it survives into the archive's usage requirements - BUILD_INTERFACE keeps it
# out of the export set, which would otherwise demand VMA be exported too.
target_link_libraries(NoorRHI
    PUBLIC Vulkan::Vulkan
    PRIVATE $<BUILD_INTERFACE:GPUOpen::VulkanMemoryAllocator>)

# What every shader built for NoorRHI passes to slangc: noorrhi.slang on the
# import path, its bindless set as set 0, and SPIR-V emitted directly. Cached so
# a parent project that adds this directory sees it too.
set(NOORRHI_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders" CACHE INTERNAL "")
set(NOORRHI_SLANG_FLAGS -I "${NOORRHI_SHADER_DIR}" -bindless-space-index 0
    -emit-spirv-directly CACHE INTERNAL "")

# Examples and tests compile Slang shaders. An unset VULKAN_SDK must not become
# a "/bin" hint, which would prefer the distribution's older tools.
set(NOORRHI_VULKAN_SDK_BIN_HINTS "")
if(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
    list(APPEND NOORRHI_VULKAN_SDK_BIN_HINTS "$ENV{VULKAN_SDK}/bin")
endif()

if(NOORRHI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
if(NOORRHI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

install(TARGETS NoorRHI
    EXPORT NoorRHITargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(DIRECTORY include/noorrhi DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT NoorRHITargets
    FILE NoorRHITargets.cmake
    NAMESPACE NoorRHI::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NoorRHI")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NoorRHIConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/NoorRHIConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NoorRHI")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NoorRHIConfigVersion.cmake"
    VERSION ${NOORRHI_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/NoorRHIConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/NoorRHIConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NoorRHI")
