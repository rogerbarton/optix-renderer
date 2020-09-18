cmake_minimum_required(VERSION 3.1)

project(NoriDocs)

find_package(Doxygen)

if(Doxygen_FOUND)

  set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

  message(${doxyfile_in})

  add_custom_target(
    doc ALL
    COMMAND
      ${DOXYGEN_EXECUTABLE} ${doxyfile_in}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM
  )
else()
  message("Doxygen not found.")
endif()
