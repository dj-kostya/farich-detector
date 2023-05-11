cmake_minimum_required(VERSION 3.25)
project(detector_cpp)

set(CMAKE_CXX_STANDARD 20)
add_executable(detector_cpp main.cpp src/geometry/Plane.cpp src/geometry/Ellipse.cpp src/geometry/Ellipse.h src/filters/FilterByTime.cpp src/filters/FilterByTime.h src/filters/FilterByTime.h src/filters/FilterInPlane.cpp src/filters/FilterInPlane.h src/filters/FilterInPlane.h src/geometry/Point.cpp src/geometry/Point.h src/geometry/Point.h src/filters/IFilter.h)
find_package(DataFrame CONFIG REQUIRED)
target_link_libraries(detector_cpp PRIVATE DataFrame::DataFrame)
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(detector_cpp PRIVATE opencv_ml tegra_hal opencv_dnn opencv_core)
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(detector_cpp PRIVATE Eigen3::Eigen)

