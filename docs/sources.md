# Sources and verification boundary

Consulted for this planning pass on 2026-08-10. The user has authorized all
listed datasets for personal, non-commercial research. The ledger therefore
records supported claims and technical/semantic constraints rather than
licence gates. Complete moderate sources and bounded deterministic subsamples of
large sources are both valid packets. Prefer pinned releases/commits at implementation time because
`latest` documentation can change.

## ESP32-P4, camera, and deployment

- [Waveshare ESP32-P4-Module-DEV-KIT documentation](https://www.waveshare.com/wiki/ESP32-P4-Module-DEV-KIT)
- [Espressif OV5647 driver source](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_cam_sensor/sensors/ov5647/ov5647.c)
- [OV5647 Kconfig modes](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_cam_sensor/sensors/ov5647/Kconfig.ov5647)
- [ESP32-P4 ISP API](https://docs.espressif.com/projects/esp-idf/en/latest/esp32p4/api-reference/peripherals/isp.html)
- [ESP32-P4 PPA API](https://docs.espressif.com/projects/esp-idf/en/latest/esp32p4/api-reference/peripherals/ppa.html)
- [ESP32-P4 camera-controller buffer/DMA API](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/camera_driver.html)
- [Current esp-video CSI format implementation](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_video/src/device/esp_video_csi_format.c)
- [esp-video format README](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_video/README.md)
- [esp-video Kconfig and backup-buffer requirement](https://github.com/espressif/esp-video-components/blob/6d4d1fbe1fccd5306cf93a0e3f7e106e6c200f8e/esp_video/Kconfig)
- [ESP32-P4 PPA low-level revision guard](https://github.com/espressif/esp-idf/blob/v6.0.2/components/esp_hal_ppa/esp32p4/include/hal/ppa_ll.h)
- [ESP-IDF chip revision guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32p4/api-reference/system/chip_revision.html)
- [ESP-IDF external RAM guidance](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/external-ram.html)
- [ESP-IDF build system](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/build-system.html)
- [ESP32-P4 JTAG debugging](https://docs.espressif.com/projects/esp-idf/en/latest/esp32p4/api-guides/jtag-debugging/index.html)
- [ESP-IDF speed guidance](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/performance/speed.html)
- [ESP-IDF binary-size guidance](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/performance/size.html)
- [ESP-IDF compiler/build Kconfig reference](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/kconfig-reference.html)

## ESP-IDF and VS Code setup

- [ESP-IDF v6.0.2 macOS installation](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/get-started/macos-setup.html)
- [EIM command-line installation](https://docs.espressif.com/projects/idf-im-ui/en/latest/cli_installation.html)
- [EIM CLI commands](https://docs.espressif.com/projects/idf-im-ui/en/latest/cli_commands.html)
- [ESP-IDF extension installation](https://docs.espressif.com/projects/vscode-esp-idf-extension/en/latest/installation.html)
- [ESP-IDF extension commands](https://docs.espressif.com/projects/vscode-esp-idf-extension/en/latest/commands.html)
- [ESP32-P4 project build/flash/monitor](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/get-started/start-project.html)
- [ESP32-P4 RAM-use guidance](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-guides/performance/ram-usage.html)

## Training and quantization

- [ESP-DL repository and current version notes](https://github.com/espressif/esp-dl)
- [ESP-DL quantization guide and P4 rules](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html)
- [ESP-DL TQT guide](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/quantize_model_with_TQT.html)
- [ESP-DL AutoQuant guide](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/auto_quantization/how_to_use_AutoQuant.html)
- [ESP-DL YOLO11n-pose PTQ/QAT example](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_yolo11n-pose.html)
- [ESP-DL model testing and memory profiling](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_load_test_profile_model.html)
- [ESP-DL input quantization and runtime contract](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_run_model.html)
- [ESP-DL project organization](https://docs.espressif.com/projects/esp-dl/en/latest/introduction/esp_dl_project.html)
- [CenterNet: Objects as Points](https://arxiv.org/abs/1904.07850)
- [Trained Quantization Thresholds paper](https://arxiv.org/abs/1903.08066)
- [OpenCV fisheye model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)

## Dataset evidence ledger

| Source | Exact claim supported | Primary evidence | Research-use note |
|---|---|---|---|
| CrowdHuman | head, visible/full body, ignore annotations | [download page](https://www.crowdhuman.org/download.html) | complete packet if within cap; preserve head/body/ignore semantics |
| RPEE-Heads | distant/dense visible-head benchmark across recordings | [paper](https://arxiv.org/abs/2411.18164), [data record](https://juser.fz-juelich.de/record/1041726/files/RPEE-Heads_Benchmark_A_Dataset_and_Empirical_Comparison_of_Deep_Learning_Algorithms_for_Pedestrian_Head_Detection_in_Crowds-1.pdf?version=1) | complete packet if within cap; preserve recording split |
| JRDB-Pose | robot sequences with head/body/pose/occlusion semantics | [dataset page](https://jrdb.erc.monash.edu.au/dataset/) | sequence-stratified packet |
| SCUT-HEAD | classroom and internet full-head boxes | [repository](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release) | complete packet if within cap; preserve full-head semantics |
| Open Images V7 | exact class boxes, verification, group/coverage attributes | [downloads/files](https://storage.googleapis.com/openimages/web/download_v7.html), [coverage facts](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) | deterministic class/coverage subset; parse exact MIDs and attributes |
| COCO 2017 | auxiliary person boxes/masks/keypoints and image metadata | [official site](https://cocodataset.org/) | diverse bounded auxiliary packet |
| JHU-CROWD++ | head points/approximate geometry, weather/light diversity | [paper](https://arxiv.org/abs/2004.03597) | scale/weather/light-stratified packet |
| NWPU-Crowd | extreme-density crowd points/boxes | [paper/project](https://arxiv.org/abs/2001.03360) | bounded extreme-density packet |
| HollywoodHeads | movie-frame head boxes | [project](https://www.di.ens.fr/willow/research/headdetection/) | movie-grouped bounded packet |
| WIDER FACE | face boxes across scale/pose/events | [paper](https://arxiv.org/abs/1511.06523) | event/scale-stratified auxiliary packet |
| DARK FACE | real low-light face annotations | [challenge page](https://flyywh.github.io/CVPRW2019LowLight/) | bounded low-light packet |
| UFDD | unconstrained face-detection stress evidence | [paper](https://arxiv.org/abs/1804.10275) | bounded stress packet |
| OCHuman | heavily occluded body masks/pose | [repository](https://github.com/liruilong940607/OCHumanApi) | bounded occlusion packet |
| CrowdPose | crowded body keypoints | [repository](https://github.com/jeffffffli/CrowdPose) | bounded pose packet |
| WiderPerson | diverse person boxes | [paper](https://arxiv.org/abs/1909.12118) | bounded person/context packet |
| FishEye8K | native fisheye person boxes from 18 cameras | [repository](https://github.com/MoyoG/FishEye8K), [paper](https://arxiv.org/abs/2305.17449) | camera-stratified native-fisheye packet |
| NightOwls | night pedestrian annotations with size/ignore semantics | [annotation examples](https://www.nightowls-dataset.org/examples/), [details](https://www.nightowls-dataset.org/about/), [download](https://www.nightowls-dataset.org/download/) | bounded partial retrieval only; never fetch roughly 285 GB for a small retained subset |
| WoodScape | automotive native-fisheye masks/boxes | [repository](https://github.com/valeoai/woodscape) | camera/scene-stratified native-fisheye packet |

## Important unresolved facts

- The fixed board's ESP32-P4 silicon revision and connected cable have not been
  inspected.
- RAW10→YUV420→small GRAY8 is a documented/source-supported candidate, not a
  validated end-to-end board result.
- Individual dataset hosts may still require login, manual acceptance, or retry;
  those are operational blockers to record. A very large source also requires a
  partial-retrieval path before it enters the bounded corpus.
- Product thresholds for accuracy, false positives, memory, latency, and FPS
  remain to be declared at the entry/freeze boundaries in the controlling plan.
- No dataset has been downloaded and no new implementation has begun under this
  rebuild plan.
