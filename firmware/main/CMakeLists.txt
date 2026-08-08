set(TRACKER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

idf_component_register(
    SRCS
        "src/tracker_benchmark_main.c"
        "src/rv32/tracker_argmax_hwc16_rv32.S"
        "src/rv32/tracker_perf_rv32.S"
        "${TRACKER_ROOT}/src/core/alpha_beta.c"
        "${TRACKER_ROOT}/src/core/decode.c"
        "${TRACKER_ROOT}/src/core/mailbox.c"
        "${TRACKER_ROOT}/src/core/preprocess.c"
        "${TRACKER_ROOT}/src/core/tensor.c"
    INCLUDE_DIRS
        "include"
        "${TRACKER_ROOT}/include"
    PRIV_REQUIRES
        esp_hw_support
        esp_system
        heap
)

set_source_files_properties(
    "src/rv32/tracker_argmax_hwc16_rv32.S"
    "src/rv32/tracker_perf_rv32.S"
    PROPERTIES COMPILE_OPTIONS "-Wa,--fatal-warnings"
)
