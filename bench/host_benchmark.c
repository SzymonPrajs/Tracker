#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HEATMAP_WIDTH 72u
#define HEATMAP_HEIGHT 40u
#define DEFAULT_ITERATIONS 2000u
#define DEFAULT_WARMUP 100u

static volatile uint64_t observable_checksum;

typedef struct {
    uint32_t x_q16;
    uint32_t y_q16;
    uint64_t weight;
    int valid;
} centroid_t;

static uint64_t monotonic_ns(void)
{
    struct timespec value;
#ifdef CLOCK_MONOTONIC_RAW
    const clockid_t clock_id = CLOCK_MONOTONIC_RAW;
#else
    const clockid_t clock_id = CLOCK_MONOTONIC;
#endif
    if (clock_gettime(clock_id, &value) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static uint32_t xorshift32(uint32_t *state)
{
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static void make_heatmap(uint16_t *data, uint32_t seed)
{
    uint32_t state = seed ? seed : UINT32_C(1);
    for (size_t index = 0; index < HEATMAP_WIDTH * HEATMAP_HEIGHT; ++index) {
        data[index] = (uint16_t)(xorshift32(&state) & UINT32_C(0x03ff));
    }

    /* A deterministic dominant target ensures that the result is non-trivial. */
    data[17u * HEATMAP_WIDTH + 43u] = UINT16_MAX;
}

static centroid_t centroid_q16(const uint16_t *data, size_t width, size_t height)
{
    uint64_t weight = 0;
    uint64_t weighted_x = 0;
    uint64_t weighted_y = 0;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            uint64_t sample = data[y * width + x];
            weight += sample;
            weighted_x += sample * x;
            weighted_y += sample * y;
        }
    }

    if (weight == 0) {
        return (centroid_t){0, 0, 0, 0};
    }

    return (centroid_t){
        .x_q16 = (uint32_t)((weighted_x << 16) / weight),
        .y_q16 = (uint32_t)((weighted_y << 16) / weight),
        .weight = weight,
        .valid = 1,
    };
}

static int compare_u64(const void *left, const void *right)
{
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint64_t percentile(const uint64_t *sorted, size_t count, unsigned percent)
{
    size_t rank = ((size_t)percent * count + 99u) / 100u;
    if (rank == 0) {
        rank = 1;
    }
    return sorted[rank - 1u];
}

static uint64_t parse_unsigned(const char *text, const char *name, int allow_zero)
{
    char *end = NULL;
    errno = 0;
    unsigned long long value;
    if (text[0] == '-') {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || (!allow_zero && value == 0)) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static int verify_known_fixture(centroid_t *actual)
{
    uint16_t fixture[HEATMAP_WIDTH * HEATMAP_HEIGHT] = {0};
    centroid_t zero = centroid_q16(fixture, HEATMAP_WIDTH, HEATMAP_HEIGHT);
    fixture[17u * HEATMAP_WIDTH + 43u] = UINT16_MAX;
    *actual = centroid_q16(fixture, HEATMAP_WIDTH, HEATMAP_HEIGHT);
    return !zero.valid && actual->valid && actual->weight == UINT16_MAX &&
           actual->x_q16 == (43u << 16) && actual->y_q16 == (17u << 16);
}

int main(int argc, char **argv)
{
    char run_id[80];
    size_t iterations = DEFAULT_ITERATIONS;
    size_t warmup = DEFAULT_WARMUP;
    uint32_t seed = UINT32_C(0x5eed1234);

    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--iterations") == 0 && index + 1 < argc) {
            uint64_t value = parse_unsigned(argv[++index], "iteration count", 0);
            if (value > SIZE_MAX / sizeof(uint64_t)) {
                fputs("iteration count is too large\n", stderr);
                return EXIT_FAILURE;
            }
            iterations = (size_t)value;
        } else if (strcmp(argv[index], "--warmup") == 0 && index + 1 < argc) {
            uint64_t value = parse_unsigned(argv[++index], "warmup count", 1);
            if (value > SIZE_MAX) {
                fputs("warmup count is too large\n", stderr);
                return EXIT_FAILURE;
            }
            warmup = (size_t)value;
        } else if (strcmp(argv[index], "--seed") == 0 && index + 1 < argc) {
            uint64_t value = parse_unsigned(argv[++index], "seed", 1);
            if (value > UINT32_MAX) {
                fputs("seed is too large\n", stderr);
                return EXIT_FAILURE;
            }
            seed = (uint32_t)value;
        } else {
            fprintf(stderr, "usage: %s [--iterations N] [--warmup N] [--seed N]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    uint16_t *heatmap = malloc(sizeof(*heatmap) * HEATMAP_WIDTH * HEATMAP_HEIGHT);
    uint64_t *samples = malloc(sizeof(*samples) * iterations);
    if (heatmap == NULL || samples == NULL) {
        fputs("allocation failed\n", stderr);
        free(samples);
        free(heatmap);
        return EXIT_FAILURE;
    }

    make_heatmap(heatmap, seed);
    centroid_t benchmark_result = centroid_q16(heatmap, HEATMAP_WIDTH, HEATMAP_HEIGHT);
    if (!benchmark_result.valid) {
        fputs("generated fixture has no centroid\n", stderr);
        free(samples);
        free(heatmap);
        return EXIT_FAILURE;
    }

    for (size_t index = 0; index < warmup; ++index) {
        centroid_t result = centroid_q16(heatmap, HEATMAP_WIDTH, HEATMAP_HEIGHT);
        observable_checksum ^= result.weight + result.x_q16 + result.y_q16;
    }

    uint64_t measurement_start_ns = monotonic_ns();
    for (size_t index = 0; index < iterations; ++index) {
        uint64_t start = monotonic_ns();
        centroid_t result = centroid_q16(heatmap, HEATMAP_WIDTH, HEATMAP_HEIGHT);
        uint64_t end = monotonic_ns();
        samples[index] = end - start;
        observable_checksum ^= result.weight + result.x_q16 + result.y_q16 + index;
    }
    uint64_t measurement_end_ns = monotonic_ns();

    centroid_t known_actual;
    int correct = verify_known_fixture(&known_actual);

    qsort(samples, iterations, sizeof(*samples), compare_u64);
    uint64_t sum_ns = 0;
    for (size_t index = 0; index < iterations; ++index) {
        sum_ns += samples[index];
    }
    if (measurement_end_ns == measurement_start_ns) {
        fputs("measurement interval was zero\n", stderr);
        free(samples);
        free(heatmap);
        return EXIT_FAILURE;
    }
    if (snprintf(run_id, sizeof(run_id), "host-centroid-%08" PRIx32 "-%" PRIu64,
                 seed, measurement_start_ns) >= (int)sizeof(run_id)) {
        fputs("run identifier overflow\n", stderr);
        free(samples);
        free(heatmap);
        return EXIT_FAILURE;
    }

    const char *compiler =
#if defined(__clang__)
        "clang";
#elif defined(__GNUC__)
        "gcc";
#else
        "unknown";
#endif

    printf("{\"schema_version\":\"benchmark-run-v1\",\"type\":\"run\","
           "\"run_id\":\"%s\","
           "\"evidence_class\":\"host_synthetic\",\"timing_scope\":\"host_postprocess_only\","
           "\"device_fps_claim\":false,\"platform\":\"native-host\","
           "\"compiler\":\"%s\",\"seed\":%" PRIu32 ",\"warmup_iterations\":%zu,"
           "\"measurement_iterations\":%zu}\n",
           run_id, compiler, seed, warmup, iterations);
    printf("{\"schema_version\":\"benchmark-run-v1\",\"type\":\"correctness\","
           "\"run_id\":\"%s\",\"name\":\"centroid_known_fixtures_q16\","
           "\"passed\":%s,\"expected\":{\"valid\":true,\"weight\":65535,"
           "\"x_q16\":2818048,\"y_q16\":1114112},"
           "\"actual\":{\"valid\":%s,\"weight\":%" PRIu64 ","
           "\"x_q16\":%" PRIu32 ",\"y_q16\":%" PRIu32 "}}\n",
           run_id, correct ? "true" : "false", known_actual.valid ? "true" : "false",
           known_actual.weight, known_actual.x_q16, known_actual.y_q16);
    printf("{\"schema_version\":\"benchmark-run-v1\",\"type\":\"summary\","
           "\"run_id\":\"%s\","
           "\"evidence_class\":\"host_synthetic\",\"timing_scope\":\"host_postprocess_only\","
           "\"device_fps_claim\":false,\"metric\":\"centroid_heatmap_q16\","
           "\"samples\":%zu,\"measurement_duration_us\":%.3f,"
           "\"latency_ns\":{\"min\":%" PRIu64 ",\"mean\":%.3f,\"p50\":%" PRIu64
           ",\"p95\":%" PRIu64 ",\"p99\":%" PRIu64 ",\"max\":%" PRIu64 "},"
           "\"host_iterations_per_second\":%.3f,\"observable_checksum\":%" PRIu64 "}\n",
           run_id, iterations, (double)(measurement_end_ns - measurement_start_ns) / 1000.0,
           samples[0], (double)sum_ns / (double)iterations, percentile(samples, iterations, 50),
           percentile(samples, iterations, 95), percentile(samples, iterations, 99),
           samples[iterations - 1u],
           (double)iterations * 1000000000.0 / (double)(measurement_end_ns - measurement_start_ns),
           observable_checksum);

    free(samples);
    free(heatmap);
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
