cmake_minimum_required(VERSION 3.22)
project(tracker C)

add_library(tracker src/tracker.c)
target_include_directories(tracker PUBLIC include)

add_executable(test_tracker tests/test_tracker.c)
target_link_libraries(test_tracker PRIVATE tracker)

enable_testing()
add_test(NAME tracker COMMAND test_tracker)
