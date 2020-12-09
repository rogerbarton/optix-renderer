# This is the main CMake build file used to compile Nori
cmake_minimum_required(VERSION 3.12)
project(nori)

option(NORI_USE_GUI "Build the GUI" ON)
option(NORI_USE_IMGUI "Use ImGui instead of NanoGui, warptest always uses NanoGUI" ON)
option(NORI_USE_VOLUMES "Enable Volumes" ON)
option(NORI_BUILD_DOCS "Build Nori Docs" OFF)
option(NORI_USE_OPTIX "Enable OptiX, requires CUDA" ON)

# This file included below takes care of compiling dependencies
# and setting compiler flags on different platforms
include("CMakeConfig.txt")

include_directories(ext)

if(NORI_BUILD_DOCS)
  add_subdirectory(docs)
endif(NORI_BUILD_DOCS)

# The following lines build the main executable. If you add a source
# code file to Nori, be sure to include it in this list.
add_executable(nori

    # Header files
    include/nori/bbox.h
    include/nori/bitmap.h
    include/nori/block.h
    include/nori/bsdf.h
    include/nori/bvh.h
    include/nori/camera.h
    include/nori/perspective.h
    include/nori/color.h
    include/nori/common.h
    include/nori/dpdf.h
    include/nori/frame.h
    include/nori/integrator.h
    include/nori/emitter.h
    include/nori/kdtree.h
    include/nori/mesh.h
    include/nori/sphere.h
    include/nori/object.h
    include/nori/parser.h
    include/nori/proplist.h
    include/nori/photon.h
    include/nori/ray.h
    include/nori/render.h
    include/nori/rfilter.h
    include/nori/sampler.h
    include/nori/scene.h
    include/nori/shape.h
    include/nori/texture.h
    include/nori/timer.h
    include/nori/transform.h
    include/nori/vector.h
    include/nori/warp.h
    include/nori/HDRLoader.h
    include/nori/denoiser.h
    include/nori/medium.h
    include/nori/phase.h

    # Source code files
    src/utils/bitmap.cpp
    src/utils/block.cpp
    src/utils/bvh.cpp
    src/utils/chi2test.cpp
    src/utils/common.cpp
    src/utils/main.cpp
    src/utils/object.cpp
    src/utils/parser.cpp
    src/utils/proplist.cpp
    src/utils/render.cpp
    src/utils/ttest.cpp
    src/utils/scene.cpp
    src/utils/warp.cpp
    src/utils/photon.cpp
    src/utils/transform.cpp

    src/textures/consttexture.cpp
    src/textures/checkerboard.cpp
    src/textures/PNGTexture.cpp

    src/bsdf/diffuse.cpp
    src/bsdf/microfacet.cpp
    src/bsdf/mirror.cpp
    src/bsdf/dielectric.cpp
    src/bsdf/disney.cpp
    src/bsdf/isophase.cpp
    src/bsdf/anisophase.cpp
    src/bsdf/schlickphase.cpp

    src/integrators/photonmapper.cpp
    src/integrators/direct.cpp
    src/integrators/direct_ems.cpp
    src/integrators/direct_mats.cpp
    src/integrators/direct_mis.cpp
    src/integrators/path_mats.cpp
    src/integrators/path_mis.cpp
    src/integrators/normals.cpp
    src/integrators/av.cpp
    src/integrators/PreviewIntegrator.cpp
    src/integrators/path_vol_mats.cpp
    src/integrators/path_vol_mis.cpp
    src/integrators/EnvMapTester.cpp

    src/shapes/mesh.cpp
    src/shapes/obj.cpp
    src/shapes/shape.cpp
    src/shapes/sphere.cpp

    src/cameras/camera.cpp
    src/cameras/perspective.cpp
    src/cameras/rfilter.cpp

    src/emitters/spotlight.cpp
    src/emitters/emitter.cpp
    src/emitters/arealight.cpp
    src/emitters/pointlight.cpp
    src/emitters/environmentmap.cpp
    src/emitters/volumelight.cpp

    src/denoiser/simple.cpp

    src/samplers/adaptive.cpp
    src/samplers/independent.cpp

    src/media/medium.cpp
    src/media/vacuum.cpp
    src/media/homogmedium.cpp
    src/media/heterogmedium.cpp
    )

if(NORI_USE_VOLUMES)
  target_sources(nori PRIVATE
      include/nori/volume.h
      src/shapes/volume.vdb.cpp
      src/shapes/volume.shape.cpp
      )
endif()

add_executable(tonemapper
    include/nori/bitmap.h
    src/utils/bitmap.cpp
    src/utils/common.cpp
    src/utils/hdrToLdr.cpp)

if(NORI_USE_GUI)
  # Append additional gui sources
  if(NORI_USE_IMGUI)
    target_sources(nori PRIVATE
        include/nori/ImguiScreen.h
        src/utils/ImguiScreen.cpp

        include/nori/ImguiHelpers.h
        src/utils/ImGuiHelpers.cpp

        include/nori/glutil.h
        src/utils/glutil.cpp
        )

  else(NORI_USE_IMGUI)
    target_sources(nori PRIVATE
        include/nori/gui.h
        src/utils/gui.cpp
        )
  endif(NORI_USE_IMGUI)

  # The following lines build the warping test application
  add_executable(warptest
      include/nori/warp.h
      src/utils/warp.cpp
      src/utils/warptest.cpp
      src/bsdf/microfacet.cpp
      src/bsdf/isophase.cpp
      src/bsdf/anisophase.cpp
      src/bsdf/schlickphase.cpp
      src/utils/object.cpp
      src/utils/proplist.cpp
      src/utils/common.cpp
      )
  add_dependencies(warptest nori)
  target_link_libraries(warptest nanogui glew ${extra_libs})
  target_compile_definitions(warptest PRIVATE -DNORI_USE_NANOGUI)
endif(NORI_USE_GUI)

if(NORI_USE_OPTIX)
  # Based on NanoVDB OptiX viewer
  cmake_policy(SET CMP0074 NEW) # find_package: Search in <PackageName>_ROOT variable
  set(OptiX_ROOT ${CMAKE_SOURCE_DIR}/ext/optix)
  find_package(CUDA)
  find_package(OptiX)

  if(CUDA_FOUND AND OptiX_FOUND)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(NANOVDB_CUDA_EXTENDED_LAMBDA "--expt-extended-lambda")
    if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
      set(NANOVDB_CUDA_EXTENDED_LAMBDA "--extended-lambda")
    endif()

    set(CMAKE_CUDA_FLAGS
        "${NANOVDB_CUDA_EXTENDED_LAMBDA} -use_fast_math -lineinfo")

    # find NVRTC library.
    find_library(
        CUDA_nvrtc_LIBRARY nvrtc "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
        "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32")

    # workaround for win32 bug when nvcc "--keep" is used.
    if(WIN32)
      if(NANOVDB_CUDA_KEEP_PTX)
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/x64/Release")
        set(CMAKE_CUDA_FLAGS_RELEASE
            " --source-in-ptx --keep ${CMAKE_CUDA_FLAGS_RELEASE}")
      endif()
    endif()

    enable_language(CUDA)

    target_include_directories(nori PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${OptiX_INCLUDE_DIR}
        )

    target_compile_definitions(nori PRIVATE
        "-DNORI_USE_OPTIX"
        "-DNORI_NVRTC_OPTIX_DIR=\"${OptiX_INCLUDE_DIR}\""
        "-DNORI_NVRTC_CUDA_DIR=\"${CUDA_INCLUDE_DIRECTORY}\"")

    target_link_libraries(nori
        ${CUDA_LIBRARIES}
        ${CUDA_nvrtc_LIBRARY}
        # OptiX is header only
        )

    set(NORI_CU_FILES
        include/nori/optix/cuda/raygen.cpp
        include/nori/optix/cuda/shading.cpp
        include/nori/optix/cuda/geometry.cpp
        include/nori/optix/cuda/RadiancePrd.h

        include/nori/optix/cuda_shared/LaunchParams.h
        include/nori/optix/cuda_shared/RayParams.h
        include/nori/optix/cuda_shared/GeometryData.h
        include/nori/optix/cuda_shared/BsdfData.h
        include/nori/optix/cuda_shared/MediumData.h
        include/nori/optix/cuda_shared/EmitterData.h

        include/nori/optix/cuda/sutil/helpers.h
        include/nori/optix/cuda/sutil/vec_math.h
        include/nori/optix/cuda/sutil/random.h
        include/nori/optix/cuda/sutil/warp.h
        )

    target_sources(nori PRIVATE
        include/nori/optix/OptixRenderer.h
        include/nori/optix/OptixRenderer.cpp

        include/nori/optix/OptixState.h
        include/nori/optix/OptixState.cpp
        include/nori/optix/OptixState.rtc.cpp
        include/nori/optix/OptixState.as.cpp
        include/nori/optix/OptixState.render.cpp

        include/nori/optix/OptixSbtTypes.h

        include/nori/optix/sutil/Exception.h
        include/nori/optix/sutil/CUDAOutputBuffer.h
        include/nori/optix/sutil/host_vec_math.h

        ${NORI_CU_FILES}
        )

    # Don't compile cu files, nvrtc will be used
    set_source_files_properties(${NORI_CU_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
    set_source_files_properties(${NORI_CU_FILES} PROPERTIES LANGUAGE CUDA)

    message(STATUS "Nori uses OptiX")
  endif()
endif()

# -- Add C++ defines
# The following C++ macros may be defined:
# DISABLE_NORI_GUI
# NORI_USE_IMGUI
# NORI_USE_NANOGUI
# NORI_USE_VOLUMES

if(NOT NORI_USE_GUI)
  message(STATUS "NORI_USE_GUI is off")

  # Disable flag for preprocessor
  target_compile_definitions(nori PRIVATE -DDISABLE_NORI_GUI)
else()
  if(NORI_USE_IMGUI)
    target_compile_definitions(nori PRIVATE -DNORI_USE_IMGUI)
    message(STATUS "Nori uses ImGui")
  else()
    target_compile_definitions(nori PRIVATE -DNORI_USE_NANOGUI)
    message(STATUS "Nori uses Nanogui")
  endif()
endif()

if(NORI_USE_VOLUMES)
  message(STATUS "Nori uses Volumes")
  target_compile_definitions(nori PUBLIC -DNORI_USE_VOLUMES)
else()
  message(STATUS "Nori doesn't use Volumes")
endif()


# Nori depends on some libraries created in CMakeConfig.txt. The following two
# lines ensure that Nori is built *after* those libraries have been created.
#add_dependencies(nori OpenEXR_p)
#add_dependencies(nori tbb_p)
add_dependencies(nori pugixml)
add_dependencies(nori lodepng_p)
#add_dependencies(nori openvdb_p)

add_dependencies(tonemapper nori)

# Link to dependency libraries
target_link_libraries(tonemapper ${extra_libs})
target_link_libraries(nori ${extra_libs})

set_target_properties(nori PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED TRUE
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED TRUE
    CXX_EXTENSIONS NO
    )

# vim: set et ts=2 sw=2 ft=cmake nospell:
