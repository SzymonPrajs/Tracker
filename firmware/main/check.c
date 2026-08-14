/*
 * First firmware: is this the right board, and does the camera work?
 *
 * It prints chip / flash / PSRAM, opens the MIPI-CSI camera, captures a few
 * seconds of frames, and writes a short BOARD / CAMERA summary. Keep this
 * file readable. Later tracker code and any assembly belong in other files
 * next to a C reference.
 */

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "driver/gpio.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "esp_video_device.h"
#include "esp_video_init.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "linux/videodev2.h"

#define TAG "check"
#define CAPTURE_SECONDS 3
#define BUFFER_COUNT 2
#define SCCB_SCL GPIO_NUM_8
#define SCCB_SDA GPIO_NUM_7
#define SCCB_FREQ 100000

static void print_memory(const char *when)
{
    printf("%s: internal free=%u  psram free=%u  min heap=%" PRIu32 "\n",
           when,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           esp_get_minimum_free_heap_size());
}

static bool print_board(void)
{
    esp_chip_info_t chip;
    uint32_t flash_size = 0;

    esp_chip_info(&chip);
    printf("\n========== BOARD ==========\n");
    printf("target:            %s\n", CONFIG_IDF_TARGET);
    printf("CPU cores:         %d\n", chip.cores);
    printf("silicon revision:  v%d.%d\n", chip.revision / 100, chip.revision % 100);

    if (esp_flash_get_size(NULL, &flash_size) == ESP_OK) {
        printf("flash:             %" PRIu32 " MB\n", flash_size / (1024U * 1024U));
    } else {
        printf("flash:             unread\n");
    }

    if (esp_psram_is_initialized()) {
        printf("PSRAM:             %u MB, initialized\n",
               (unsigned)(esp_psram_get_size() / (1024U * 1024U)));
    } else {
        printf("PSRAM:             NOT initialized\n");
        return false;
    }

    print_memory("after boot");
    printf("expected: esp32p4, 2 cores, ~16 MB flash, ~32 MB PSRAM\n");
    return true;
}

static esp_err_t init_camera(void)
{
    const esp_video_init_csi_config_t csi = {
        .sccb_config = {
            .init_sccb = true,
            .i2c_config = {
                .port = 0,
                .scl_pin = SCCB_SCL,
                .sda_pin = SCCB_SDA,
            },
            .freq = SCCB_FREQ,
        },
        .reset_pin = GPIO_NUM_NC,
        .pwdn_pin = GPIO_NUM_NC,
    };
    const esp_video_init_config_t config = {
        .csi = &csi,
    };

    printf("\n========== CAMERA ==========\n");
    printf("MIPI-CSI SCCB: SCL=GPIO%d SDA=GPIO%d\n", SCCB_SCL, SCCB_SDA);
    return esp_video_init(&config);
}

static bool capture_default_stream(void)
{
    const int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    struct v4l2_capability capability;
    struct v4l2_format format = {.type = type};
    struct v4l2_requestbuffers req;
    struct v4l2_buffer buf;
    uint8_t *buffers[BUFFER_COUNT] = {0};
    int fd = open(ESP_VIDEO_MIPI_CSI_DEVICE_NAME, O_RDONLY);
    uint32_t frames = 0;
    uint32_t bytes = 0;

    if (fd < 0) {
        ESP_LOGE(TAG, "could not open %s", ESP_VIDEO_MIPI_CSI_DEVICE_NAME);
        return false;
    }

    if (ioctl(fd, VIDIOC_QUERYCAP, &capability) != 0) {
        ESP_LOGE(TAG, "VIDIOC_QUERYCAP failed");
        close(fd);
        return false;
    }
    printf("driver:            %s\n", capability.driver);
    printf("card:              %s\n", capability.card);
    printf("bus:               %s\n", capability.bus_info);

    if (ioctl(fd, VIDIOC_G_FMT, &format) != 0) {
        ESP_LOGE(TAG, "VIDIOC_G_FMT failed");
        close(fd);
        return false;
    }
    printf("format fourcc:     %.4s\n", (char *)&format.fmt.pix.pixelformat);
    printf("frame size:        %" PRIu32 " x %" PRIu32 "\n",
           format.fmt.pix.width, format.fmt.pix.height);

    memset(&req, 0, sizeof(req));
    req.count = BUFFER_COUNT;
    req.type = type;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) != 0) {
        ESP_LOGE(TAG, "VIDIOC_REQBUFS failed");
        close(fd);
        return false;
    }

    for (int i = 0; i < BUFFER_COUNT; i++) {
        memset(&buf, 0, sizeof(buf));
        buf.type = type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) != 0) {
            ESP_LOGE(TAG, "VIDIOC_QUERYBUF failed");
            close(fd);
            return false;
        }
        buffers[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        if (!buffers[i]) {
            ESP_LOGE(TAG, "mmap failed");
            close(fd);
            return false;
        }
        if (ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
            ESP_LOGE(TAG, "VIDIOC_QBUF failed");
            close(fd);
            return false;
        }
    }

    if (ioctl(fd, VIDIOC_STREAMON, &type) != 0) {
        ESP_LOGE(TAG, "VIDIOC_STREAMON failed");
        close(fd);
        return false;
    }

    int64_t start = esp_timer_get_time();
    while (esp_timer_get_time() - start < CAPTURE_SECONDS * 1000000LL) {
        memset(&buf, 0, sizeof(buf));
        buf.type = type;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd, VIDIOC_DQBUF, &buf) != 0) {
            ESP_LOGE(TAG, "VIDIOC_DQBUF failed");
            ioctl(fd, VIDIOC_STREAMOFF, &type);
            close(fd);
            return false;
        }
        if (buf.flags & V4L2_BUF_FLAG_DONE) {
            frames++;
            bytes += buf.bytesused;
        }
        if (ioctl(fd, VIDIOC_QBUF, &buf) != 0) {
            ESP_LOGE(TAG, "VIDIOC_QBUF failed");
            ioctl(fd, VIDIOC_STREAMOFF, &type);
            close(fd);
            return false;
        }
    }

    ioctl(fd, VIDIOC_STREAMOFF, &type);
    close(fd);

    printf("captured:          %" PRIu32 " frames in %d s\n", frames, CAPTURE_SECONDS);
    if (frames > 0) {
        printf("bytes / frame:     %" PRIu32 "\n", bytes / frames);
        printf("FPS:               %" PRIu32 "\n", frames / CAPTURE_SECONDS);
    }
    print_memory("after capture");
    return frames > 0;
}

void app_main(void)
{
    bool board_ok = print_board();
    bool camera_ok = false;

    if (init_camera() != ESP_OK) {
        printf("camera init failed. Check the CSI ribbon (not DSI), power, and that\n");
        printf("the log above mentions OV5647 / PID=0x5647.\n");
    } else if (!capture_default_stream()) {
        printf("camera opened but no frames arrived.\n");
    } else {
        camera_ok = true;
    }

    printf("\n========== SUMMARY ==========\n");
    printf("BOARD:   %s\n", board_ok ? "OK" : "FAIL");
    printf("CAMERA:  %s\n", camera_ok ? "OK" : "FAIL");
    printf("Save this log with: idf.py -p PORT flash monitor | tee hardware-check.log\n");
    printf("Quit the monitor with Ctrl+]\n\n");

    while (true) {
        ESP_LOGI(TAG, "alive  internal=%u  psram=%u",
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        vTaskDelay(pdMS_TO_TICKS(5000));
    }
}
