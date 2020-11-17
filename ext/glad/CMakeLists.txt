project(glad C)

find_package(OpenGL REQUIRED)

add_library(glad SHARED
    src/glad.c
    include/glad/glad.h
    include/KHR/khrplatform.h
    )

target_compile_definitions(glad
    PRIVATE GLAD_GLAPI_EXPORT GLAD_GLAPI_EXPORT_BUILD
    PUBLIC GLAD_GLAPI_EXPORT
    )

target_include_directories(glad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

target_link_libraries(glad PUBLIC
    ${OPENGL_LIBRARIES}
    ${CMAKE_DL_LIBS}
    )

if(WIN32)
  # Workaround for Windows not looking in subdirectories for .dlls
  get_target_property(target_type glad TYPE)
  if(target_type STREQUAL SHARED_LIBRARY)
    add_custom_command(
        TARGET glad POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_BINARY_DIR})
  endif()
endif()