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
  src/emitter.cpp
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
  src/adaptive.cpp
  src/camera.cpp
  src/transform.cpp
  src/PreviewIntegrator.cpp
  src/isophase.cpp
  src/anisophase.cpp
)

if(NORI_USE_VOLUMES)
  target_sources(nori PRIVATE
    include/nori/volume.h
    src/volume.vdb.cpp
    src/volume.shape.cpp
  )
endif()

add_executable(tonemapper
        include/nori/bitmap.h
        src/bitmap.cpp
        src/common.cpp
        src/hdrToLdr.cpp)

if(NORI_USE_GUI)
  # Append additional gui sources
  if(NORI_USE_IMGUI)
    target_sources(nori PRIVATE
      include/nori/ImguiScreen.h
      src/ImguiScreen.cpp

      include/nori/ImguiHelpers.h
      src/ImGuiHelpers.cpp

      include/nori/glutil.h
      src/glutil.cpp
    )

  else(NORI_USE_IMGUI)
    target_sources(nori PRIVATE
      include/nori/gui.h
      src/gui.cpp
    )
  endif(NORI_USE_IMGUI)

  # The following lines build the warping test application
  add_executable(warptest
      include/nori/warp.h
      src/warp.cpp
      src/warptest.cpp
      src/microfacet.cpp
      src/isophase.cpp
      src/anisophase.cpp
      src/object.cpp
      src/proplist.cpp
      src/common.cpp
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
  message(STATUS "Make sure to use the right gcc version:")
  message(STATUS "module load new gcc/4.9.2 cmake/3.13.5")

  # Disable flag for preprocessor
  compile_definitions(-DDISABLE_NORI_GUI -DEIGEN_DONT_VECTORIZE -DEIGEN_DISABLE_UNALIGNED_ARRAY_ACCESS)
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
