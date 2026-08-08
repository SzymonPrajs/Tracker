cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(tracker_host_benchmark C)
  include(CTest)
endif()

add_executable(tracker_host_benchmark host_benchmark.c)
target_compile_features(tracker_host_benchmark PRIVATE c_std_11)

if(MSVC)
  target_compile_options(tracker_host_benchmark PRIVATE /W4 /WX)
else()
  target_compile_options(tracker_host_benchmark PRIVATE -O2 -Wall -Wextra -Wpedantic -Werror)
endif()

if(NOT DEFINED BUILD_TESTING OR BUILD_TESTING)
  add_test(NAME tracker_host_benchmark_smoke COMMAND tracker_host_benchmark --iterations 100 --warmup 5)
endif()
