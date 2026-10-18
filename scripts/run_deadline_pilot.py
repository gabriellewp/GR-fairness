import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from additional_baselines.deadline_pilot import prepare, run_case

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("prepare", "case"))
parser.add_argument("--manifest", required=True)
parser.add_argument("--output-dir")
parser.add_argument("--index", type=int)
args = parser.parse_args()
if args.phase == "prepare":
    print(prepare(args.manifest, args.output_dir), flush=True)
else:
    run_case(args.manifest, args.index)
