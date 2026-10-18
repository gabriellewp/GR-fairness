"""Export scoped methodology evidence; does not launch experiments.

Run with GRID-venv Python. Historical metadata is kept separate from the
current interpreter snapshot. This is not a complete project cost ledger.
"""
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from scripts.paper.archive_paths import source_root, output_root
ROOT = source_root()
OUTPUT = output_root()
OUT = OUTPUT


def reference(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    packages = sorted(({'name': d.metadata['Name'], 'version': d.version}
                       for d in importlib.metadata.distributions()),
                      key=lambda d: d['name'].lower())
    historical = []
    base = ROOT / 'GRID/outputs/beauty_5core/temporal_loo/complete_metadata/rkmeans'
    for path in sorted(base.glob('**/wandb-metadata.json')):
        data = json.loads(path.read_text())
        req = path.with_name('requirements.txt')
        historical.append({'source': reference(path),
            'metadata': {k: data[k] for k in ('os', 'python', 'gpu', 'gpu_count',
                'gpu_nvidia', 'cudaVersion') if k in data},
            'requirements_source': reference(req) if req.exists() else None,
            'requirements': req.read_text().splitlines() if req.exists() else []})
    environment = {'audit_date': '2026-10-09',
        'scope': 'Current GRID-venv plus historical Beauty RKMeans runs; not every experiment environment',
        'current': {'python': platform.python_version(), 'platform': platform.platform(),
                    'os_release': Path('/etc/os-release').read_text(), 'packages': packages},
        'historical_runs': historical,
        'limitations': ['Current environment is not proof of historical versions.',
            'W&B cudaVersion is not the PyTorch build CUDA version.',
            'Hardware capacity and allocation are not measured peak usage.']}
    (OUT / 'METHODOLOGY_ENVIRONMENT.json').write_text(json.dumps(environment, indent=2)+'\n')
    evidence = {'audit_date': '2026-10-09', 'scope': 'Saved continuation configurations and reranking sweeps; original-paper hyperparameter provenance not independently established',
                'continuations': [], 'reranking_sweeps': []}
    artifacts = ROOT / 'additional_baselines/artifacts'
    for path in sorted(artifacts.glob('**/run_config.json')):
        data = json.loads(path.read_text())
        if 'native_schedule' in data and 'identity' in data:
            entry = {'source': reference(path), 'config': data}
            report = path.with_name('continuation.json')
            if report.exists():
                d = json.loads(report.read_text())
                entry['completion'] = {k: d.get(k) for k in ('selected_update','selection_metric','selection_policy')}
                entry['completion_source'] = reference(report)
            evidence['continuations'].append(entry)
    paths = list(artifacts.glob('*validation_tuning*/validation_tradeoff.csv'))
    paths += list(artifacts.glob('*validation_tuning*/merged/validation_tradeoff.csv'))
    for path in sorted(paths):
        rows = list(csv.DictReader(path.open()))
        manifest = path.with_name('manifest.json')
        evidence['reranking_sweeps'].append({'source': reference(path),
            'manifest': json.loads(manifest.read_text()) if manifest.exists() else None,
            'result_rows': len(rows), 'cutoffs': sorted({int(r['k']) for r in rows}),
            'observed_parameter_settings': [json.loads(p) for p in sorted({r['parameters'] for r in rows})]})
    (OUT / 'METHODOLOGY_PARAMETER_EVIDENCE.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('Wrote environment and parameter evidence:', len(historical), 'historical environments,',
          len(evidence['continuations']), 'continuation configs,', len(evidence['reranking_sweeps']), 'sweep tables')


if __name__ == '__main__':
    main()
