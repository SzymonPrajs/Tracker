#include "tracker_model_runner.h"

#include <stdint.h>
#include <vector>

#include "dl_model_base.hpp"
#include "esp_check.h"
#include "esp_log.h"
#include "tracker_model_config.h"

static const char *TAG = "tracker_model";

extern const uint8_t hcds31_int8_espdl[]
    asm("_binary_hcds31_int8_espdl_start");

extern "C" void tracker_model_test_and_profile(void)
{
    dl::Model model(
        reinterpret_cast<const char *>(hcds31_int8_espdl),
        fbs::MODEL_LOCATION_IN_FLASH_RODATA
    );
    const auto inputs = model.get_inputs();
    const auto outputs = model.get_outputs();
    ESP_ERROR_CHECK_WITHOUT_ABORT(inputs.size() == 1 ? ESP_OK : ESP_FAIL);
    ESP_ERROR_CHECK_WITHOUT_ABORT(outputs.size() == 1 ? ESP_OK : ESP_FAIL);
    if (inputs.size() != 1 || outputs.size() != 1) {
        ESP_LOGE(TAG, "expected one input and one output");
        return;
    }

    const dl::TensorBase *input = inputs.begin()->second;
    const dl::TensorBase *output = outputs.begin()->second;
    const std::vector<int> expected_input = {
        1, TRACKER_INPUT_HEIGHT, TRACKER_INPUT_WIDTH, TRACKER_INPUT_CHANNELS};
    const std::vector<int> expected_output = {
        1, TRACKER_OUTPUT_HEIGHT, TRACKER_OUTPUT_WIDTH, TRACKER_OUTPUT_CHANNELS};
    const int input_exponent = input->exponent.get();
    const int output_exponent = output->exponent.get();
    ESP_LOGI(
        TAG,
        "input exponent=%d expected=%d; output exponent=%d expected=%d",
        input_exponent,
        TRACKER_INPUT_EXPONENT,
        output_exponent,
        TRACKER_OUTPUT_EXPONENT
    );
    if (input->shape != expected_input || output->shape != expected_output ||
        input_exponent != TRACKER_INPUT_EXPONENT ||
        output_exponent != TRACKER_OUTPUT_EXPONENT) {
        ESP_LOGE(TAG, "model/header shape or quantization contract mismatch");
        return;
    }

    ESP_ERROR_CHECK(model.test());
    model.profile(true);
}
