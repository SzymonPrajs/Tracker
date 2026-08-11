# Data configurations

Each JSON file completely resolves one acquisition run: source adapter,
selection seed, storage envelope, byte/image caps, free-space reserve, and output
roots. Paths may be relative to the configuration file; the saved packet records
their resolved logical settings without baking local output paths into its ID.

`open_images_smoke.json` is deliberately tiny. It exercises the real official
metadata and selected-image path using four Open Images validation images. It is
evidence that acquisition works, not the project corpus and not a training set.
