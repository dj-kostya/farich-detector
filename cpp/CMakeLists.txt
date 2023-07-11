cmake_minimum_required(VERSION 3.25)
project(detector_cpp)

set(CMAKE_CXX_STANDARD 20)


add_library(geometry src/geometry/Plane.cpp src/geometry/Ellipse.cpp src/geometry/Ellipse.h src/geometry/Point.cpp src/geometry/Point.h src/geometry/Point.h)
add_library(filters src/filters/FilterByTime.cpp src/filters/FilterByTime.h src/filters/FilterByTime.h src/filters/FilterInPlane.cpp src/filters/FilterInPlane.h src/filters/FilterInPlane.h   src/filters/IFilter.h)
add_library(metrics src/metrics.cpp src/metrics.h )


add_executable(detector_cpp main.cpp src/farich/pipeline.h)

add_library(bench src/benchmarks.h)

target_compile_definitions(bench PUBLIC bench)
find_package(DataFrame CONFIG REQUIRED)
#target_link_libraries(detector_cpp PRIVATE DataFrame::DataFrame)
target_link_libraries(filters PUBLIC DataFrame::DataFrame)

find_package(OpenCV CONFIG REQUIRED)
#target_link_libraries(detector_cpp PRIVATE opencv_ml tegra_hal opencv_dnn opencv_core)
target_link_libraries(geometry PUBLIC opencv_ml tegra_hal opencv_dnn opencv_core)


find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(filters PUBLIC Eigen3::Eigen)
target_link_libraries(geometry PUBLIC Eigen3::Eigen)

find_path(RAPIDCSV_INCLUDE_DIRS "rapidcsv.h")
target_include_directories(detector_cpp PRIVATE ${RAPIDCSV_INCLUDE_DIRS})

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(metrics PRIVATE spdlog::spdlog spdlog::spdlog_header_only)
target_link_libraries(filters PRIVATE spdlog::spdlog spdlog::spdlog_header_only)
target_link_libraries(geometry PRIVATE spdlog::spdlog spdlog::spdlog_header_only)
target_link_libraries(bench PRIVATE spdlog::spdlog spdlog::spdlog_header_only)
target_link_libraries(detector_cpp PRIVATE spdlog::spdlog spdlog::spdlog_header_only)

target_link_libraries(bench PRIVATE metrics)
target_link_libraries(bench PRIVATE filters)
target_link_libraries(bench PRIVATE geometry)

target_link_libraries(detector_cpp PRIVATE metrics)
target_link_libraries(detector_cpp PRIVATE filters)
target_link_libraries(detector_cpp PRIVATE geometry)



