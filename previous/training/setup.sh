#!/usr/bin/env bash
set -eu
python3 -m venv .tools/tracker
.tools/tracker/bin/pip install -r training/requirements.txt
