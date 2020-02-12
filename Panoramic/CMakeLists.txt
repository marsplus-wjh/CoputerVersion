cmake_minimum_required(VERSION 3.13)
project(CVPro)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CVPro main.cpp HarisConer.cpp HarisConer.h Descriptor.cpp Descriptor.h Matcher.cpp Matcher.h)
target_link_libraries(CVPro ${OpenCV_LIBS})