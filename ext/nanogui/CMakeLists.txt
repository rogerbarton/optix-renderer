cmake_minimum_required (VERSION 2.8.3)

project("NanoGUI")

# Parse arguments for nested build process
if(NOT CMAKE_EXTERNAL_ARGUMENTS)
  set(CMAKE_EXTERNAL_ARGUMENTS_LIST "")
else()
  string(REPLACE "$" ";" CMAKE_EXTERNAL_ARGUMENTS "${CMAKE_EXTERNAL_ARGUMENTS}")
  set(CMAKE_EXTERNAL_ARGUMENTS_LIST ${CMAKE_EXTERNAL_ARGUMENTS})
endif()

# Run simple C converter to put font files into the data segment
add_executable(bin2c resources/bin2c.c)
set(bin2c_cmdline resources.cpp resources.h)
file(GLOB resources "${CMAKE_CURRENT_SOURCE_DIR}/resources/*.ttf")
foreach(file ${resources})
  list(APPEND bin2c_cmdline ${file})
endforeach()

include(ExternalProject)

add_custom_command(
        OUTPUT resources.cpp resources.h
        COMMAND bin2c ARGS ${bin2c_cmdline}
        DEPENDS bin2c ${resources}
        COMMENT "Running bin2c"
        PRE_BUILD VERBATIM)

# Needed to generated files
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# -- GLFW and GLEW
include_directories(${GLFW_DIR}/include)
#add_subdirectory(${GLFW_DIR})     # already done by parent cmakelists
list(APPEND nanogui_extra_libs glfw ${GLFW_LIBRARIES})

set(extra_source "")
if(WIN32)
  include_directories(${GLEW_DIR}/include)
  list(APPEND nanogui_extra_libs glew)

elseif (APPLE)
  set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")
  # Find standard libraries needed for OpenGL/GLFW
  find_library(cocoa_library Cocoa)
  find_library(opengl_library OpenGL)
  find_library(corevideo_library CoreVideo)
  find_library(iokit_library IOKit)
  list(APPEND nanogui_extra_libs ${cocoa_library} ${opengl_library} ${corevideo_library} ${iokit_library})
  set(extra_source src/darwin.mm)
elseif("${CMAKE_SYSTEM}" MATCHES "Linux")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  list(APPEND nanogui_extra_libs GL Xxf86vm Xrandr Xinerama Xcursor Xi X11 pthread dl)
endif()

include_directories()

add_library(nanogui STATIC
  ext/nanovg/src/nanovg.c resources.cpp
  include/nanogui/glutil.h src/glutil.cpp
  include/nanogui/nanogui.h src/nanogui.cpp
  include/nanogui/widget.h src/widget.cpp
  include/nanogui/theme.h src/theme.cpp
  include/nanogui/layout.h src/layout.cpp
  include/nanogui/screen.h src/screen.cpp
  include/nanogui/label.h src/label.cpp
  include/nanogui/window.h src/window.cpp
  include/nanogui/popup.h src/popup.cpp
  include/nanogui/checkbox.h src/checkbox.cpp
  include/nanogui/button.h src/button.cpp
  include/nanogui/toolbutton.h
  include/nanogui/popupbutton.h src/popupbutton.cpp
  include/nanogui/combobox.h src/combobox.cpp
  include/nanogui/progressbar.h src/progressbar.cpp
  include/nanogui/slider.h src/slider.cpp
  include/nanogui/messagedialog.h src/messagedialog.cpp
  include/nanogui/textbox.h src/textbox.cpp
  include/nanogui/imagepanel.h src/imagepanel.cpp
  include/nanogui/vscrollpanel.h src/vscrollpanel.cpp
  ${extra_source}
)

target_include_directories(nanogui PUBLIC include ext/nanovg/src ${Eigen3_DIR})
target_include_directories(nanogui PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(nanogui ${nanogui_extra_libs})

# vim: set et ts=2 sw=2 ft=cmake nospell:
