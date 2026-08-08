cmake_minimum_required(VERSION 3.22)

project(tracker_core VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(TRACKER_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ON)

add_library(tracker_core STATIC
    src/core/alpha_beta.c
    src/core/decode.c
    src/core/mailbox.c
    src/core/preprocess.c
    src/core/tensor.c
)

target_include_directories(tracker_core PUBLIC include)
target_compile_options(tracker_core PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(TRACKER_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(tracker_core PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(tracker_core PUBLIC -fsanitize=address,undefined)
endif()

enable_testing()

add_executable(tracker_core_tests tests/test_core.c)
target_link_libraries(tracker_core_tests PRIVATE tracker_core)
target_compile_options(tracker_core_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -UNDEBUG)

if(TRACKER_ENABLE_SANITIZERS AND CMAKE_C_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
    target_compile_options(tracker_core_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

add_test(NAME tracker_core_tests COMMAND tracker_core_tests)

add_subdirectory(bench)
