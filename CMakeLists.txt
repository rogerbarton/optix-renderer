# This is the main CMake build file used to compile Nori
cmake_minimum_required (VERSION 2.8)
project("nori")

option(NORI_BUILD_DOCS "Build Nori Docs" OFF)
option(NORI_USE_GUI    "Build the GUI"   ON)

if(NOT NORI_USE_GUI)
  message("NORI_USE_GUI is OFF.")

  # Disable flag for preprocessor
  add_definitions(-DDISABLE_NORI_GUI)
endif()

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
  include/nori/color.h
  include/nori/common.h
  include/nori/dpdf.h
  include/nori/frame.h
  include/nori/integrator.h
  include/nori/emitter.h
  include/nori/kdtree.h
  include/nori/mesh.h
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
  include/nori/environmentmap.h
  include/nori/HDRLoader.h
  include/nori/denoiser.h
  include/nori/volume.h

  # Source code files
  src/bitmap.cpp
  src/block.cpp
  src/bvh.cpp
  src/chi2test.cpp
  src/common.cpp
  src/consttexture.cpp
  src/checkerboard.cpp
  src/diffuse.cpp
  src/independent.cpp
  src/main.cpp
  src/mesh.cpp
  src/obj.cpp
  src/object.cpp
  src/parser.cpp
  src/perspective.cpp
  src/proplist.cpp
  src/render.cpp
  src/rfilter.cpp
  src/scene.cpp
  src/shape.cpp
  src/ttest.cpp
  src/warp.cpp
  src/microfacet.cpp
  src/photon.cpp
  src/mirror.cpp
  src/dielectric.cpp
  src/photonmapper.cpp
  src/sphere.cpp
  src/arealight.cpp
  src/normals.cpp
  src/av.cpp
  src/pointlight.cpp
  src/direct.cpp
  src/direct_ems.cpp
  src/direct_mats.cpp
  src/direct_mis.cpp
  src/path_mats.cpp
  src/path_mis.cpp
  src/disney.cpp
  src/environmentmap.cpp
  src/PNGTexture.cpp
  src/denoiser.cpp
  src/volume.cpp
)

add_executable(tonemapper
        include/nori/bitmap.h
        src/bitmap.cpp
        src/common.cpp
        src/hdrToLdr.cpp)

# Nori depends on some libraries created in CMakeConfig.txt. The following two
# lines ensure that Nori is built *after* those libraries have been created.
add_dependencies(nori OpenEXR_p)
add_dependencies(nori tbb_p)
add_dependencies(nori pugixml)
add_dependencies(nori lodepng_p)

add_dependencies(tonemapper nori)

if(NORI_USE_GUI)
  # Append additional gui sources
  target_sources(nori PRIVATE
          include/nori/gui.h
          src/gui.cpp
          )
  add_dependencies(nori nanogui_p)

  # The following lines build the warping test application
  add_executable(warptest
          include/nori/warp.h
          src/warp.cpp
          src/warptest.cpp
          src/microfacet.cpp
          src/object.cpp
          src/proplist.cpp
          src/common.cpp
          )
  add_dependencies(warptest nori)
  target_link_libraries(warptest ${extra_libs})
endif(NORI_USE_GUI)

# Link to dependency libraries
target_link_libraries(tonemapper ${extra_libs})
target_link_libraries(nori ${extra_libs})


# vim: set et ts=2 sw=2 ft=cmake nospell:
