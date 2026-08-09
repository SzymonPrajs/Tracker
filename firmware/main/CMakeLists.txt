set(ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

idf_component_register(
    SRCS
        "src/main.c"
        "src/rv32/tracker_argmax_hwc16_rv32.S"
        "src/rv32/tracker_perf_rv32.S"
        "${ROOT}/src/tracker.c"
    INCLUDE_DIRS "include" "${ROOT}/include"
)
