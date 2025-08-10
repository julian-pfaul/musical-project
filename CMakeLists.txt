cmake_minimum_required(VERSION 4.0)

project(MUSICAL_PROJECT)

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
