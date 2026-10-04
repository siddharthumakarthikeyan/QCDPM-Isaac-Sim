cmake_minimum_required(VERSION 3.18)
project(cdpr_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(cdpr STATIC src/model.cpp src/estimator.cpp src/controller.cpp src/plant.cpp src/planning.cpp)
target_include_directories(cdpr PUBLIC include)
target_link_libraries(cdpr PUBLIC Eigen3::Eigen)
target_compile_options(cdpr PRIVATE -O3 -Wall -Wextra)

# Python module (one build per interpreter: system Python for the analysis, Isaac Sim's Python for the simulation)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(cdpr_core bindings/py_cdpr.cpp)
target_link_libraries(cdpr_core PRIVATE cdpr)
install(TARGETS cdpr_core LIBRARY DESTINATION .)
