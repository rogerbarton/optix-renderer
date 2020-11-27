# This is the main CMake build file used to compile Nori
cmake_minimum_required (VERSION 2.8)
project("nori")

option(NORI_USE_GUI                "Build the GUI"               ON)
option(NORI_USE_IMGUI              "Use ImGui instead of NanoGui, warptest always uses NanoGUI"       ON)
option(NORI_USE_VOLUMES            "Enable Volumes"              ON)
option(NORI_BUILD_DOCS             "Build Nori Docs"             OFF)

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
  include/nori/HDRLoader.h
  include/nori/denoiser.h

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

  src/denoiser/denoiser.cpp

  src/samplers/adaptive.cpp
  src/samplers/independent.cpp
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
      src/utils/object.cpp
      src/utils/proplist.cpp
      src/utils/common.cpp
      )
  add_dependencies(warptest nori)
  target_link_libraries(warptest nanogui glew ${extra_libs})
  target_compile_definitions(warptest PRIVATE -DNORI_USE_NANOGUI)
endif(NORI_USE_GUI)

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
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        )

# vim: set et ts=2 sw=2 ft=cmake nospell:
