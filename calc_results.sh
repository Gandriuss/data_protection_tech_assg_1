#!/usr/bin/env bash

source .venv/bin/activate

python recovery.py
python recovery.py --run_type=bb
python recovery.py --run_type=bb_top1
python recovery.py --run_type=bb_top5