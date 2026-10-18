"""Run a restored audit against the external historical experiment workspace."""
import argparse
import importlib
import os
from pathlib import Path
import runpy
import shutil
import sys

TOOLS = {
    'assemble-comparison': ('assemble_tables.py', []),
    'assemble-rkmeans': ('assemble_rkmeans_results.py', []),
    'audit-training': ('audit_training_protocol.py', []),
    'audit-continuation': ('audit_continuation_settings.py', []),
    'audit-reranking': ('audit_reranking.py', []),
    'audit-datasets': ('audit_dataset_preprocessing.py', []),
    'audit-environment': ('audit_methodology_artifacts.py', []),
    'audit-rq1': ('audit_rq1_grid.py', ['--output-dir']),
    'parameter-inventory': ('../../document_experiment_parameters.py', []),
    'compare-source-parameters': ('../../compare_original_source_parameters.py', []),
    'compute-cost': ('../../audit_gpu_hours.py', []),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tool', choices=TOOLS)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args, remainder = parser.parse_known_args()
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    if not source.is_dir() or output.is_relative_to(source):
        parser.error('Source must exist and output must be outside the source workspace')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new or empty output directory')
    os.environ['GR_FAIRNESS_ARCHIVE_ROOT'] = str(source)
    os.environ['GR_FAIRNESS_ARCHIVE_OUTPUT'] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    name, extra = TOOLS[args.tool]
    if args.tool == 'assemble-comparison':
        # Imported renderers share the archive's inputs and this run's outputs.
        for module_name in ['scripts.build_grid_baseline_comparison', 'scripts.build_joint_fairness_comparison']:
            module = importlib.import_module(module_name)
            module.ROOT, module.OUT = source, output
            if hasattr(module, 'PAPER_CSV'):
                module.PAPER_CSV = source/'docs/rq1_grid_analysis_20261011/paper_vs_fresh_local_results.csv'
        shutil.copyfile(source/'docs/rq2_comparison_draft_20261011/pet_published_reference.json', output/'pet_published_reference.json')
    path = (Path(__file__).parent/'archive'/name).resolve()
    sys.argv = [str(path), *([*extra, str(output)] if extra else []), *remainder]
    runpy.run_path(str(path), run_name='__main__')


if __name__ == '__main__':
    main()
