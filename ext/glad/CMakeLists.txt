project(glad C)

find_package(OpenGL REQUIRED)

add_library(glad STATIC
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
