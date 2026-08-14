# Python scripts

Run scripts from the repository root:

```bash
python3 python/download.py
python3 python/preprocess.py
python3 python/train.py
python3 python/build_model.py
```

Top-level scripts are project phases. `datasets/` contains external dataset
parsers. `common/` contains code shared by more than one script. Nothing else
belongs here. `build_model.py` is the small, data-free inspection entry point for
the motion-first temporal experiment; it builds the graph and reports exact
tensor shapes, convolution MACs, parameters, and persistent state.
