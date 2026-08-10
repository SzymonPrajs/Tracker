set(ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)
set(embed_files "${ROOT}/firmware/models/hcds31-int8.espdl")

idf_component_register(
    SRCS
        "src/main.c"
        "src/model_runner.cpp"
        "src/rv32/tracker_argmax_hwc16_rv32.S"
        "src/rv32/tracker_perf_rv32.S"
        "${ROOT}/src/tracker.c"
    INCLUDE_DIRS "include" "${ROOT}/include"
    REQUIRES esp-dl
)

target_add_aligned_binary_data(${COMPONENT_LIB} ${embed_files} BINARY)
