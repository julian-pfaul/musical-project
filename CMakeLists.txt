cmake_minimum_required(VERSION 4.0)

project(MUSICAL_PROJECT)

# Example from https://docs.pytorch.org/cppdocs/installing.html
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#add_executable(example-app example-app.cpp)
#target_link_libraries(example-app "${TORCH_LIBRARIES}")
#set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)
# End of example

set(BUILD_MIDILIBRARY_ONLY ON)
set(CMAKE_POLICY_DEFAULT_CMP 2.8)
add_subdirectory(extern/midifile)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

set(CMAKE_CXX_STANDARD 23)

add_executable(midi-tools ${SOURCES} ${HEADERS})

target_include_directories(midi-tools PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(midi-tools PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/extern/midifile/include")
target_link_libraries(midi-tools PRIVATE midifile)
