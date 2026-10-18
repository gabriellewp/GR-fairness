"""Run a frozen v2 case without changing code used by active v1 cases."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from additional_baselines.deadline_pilot_v2 import run_case

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", required=True)
parser.add_argument("--index", type=int, required=True)
args = parser.parse_args()
run_case(args.manifest, args.index)
