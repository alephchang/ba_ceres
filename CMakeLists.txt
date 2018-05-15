cmake_minimum_required(VERSION 2.8.3)
project(ba_ceres)
# Use this snippet *after* PROJECT(xxx):
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

include_directories(
    ${Eigen3_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} ba_demo.cpp baproblem.cpp parametersse3.cpp inverse_depth_factor.cpp marginalization_factor.cpp utility.cpp)
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})
