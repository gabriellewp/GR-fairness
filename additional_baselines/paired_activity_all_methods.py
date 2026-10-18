"""Unified entry point for all 144 comparisons, including legacy compact COO.

Uses the completed five-method analysis and the frozen seven-trainer evaluator.
All supported score layouts are enabled from the start of a new run.
"""
import argparse
from pathlib import Path

from . import paired_activity as base
from . import paired_activity_fairdiverse as fd
from . import paired_activity_fairdiverse_retry as reader


def prepare(old_manifest, output, resamples=1000, seed=42):
    fd.prepare(old_manifest, output, resamples, seed)
    output = Path(output).resolve()
    base.write_json(output/'reader_context.json', dict(manifest=base.reference(output/'manifest.json'),
        code=base.reference(Path(reader.__file__)), wrapper_code=base.reference(Path(__file__)),
        cases=list(range(12)), reason='Complete reader: full COO, compact COO, and compact CSR'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--old-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resamples',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p=sub.add_parser('run')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--case',type=int,required=True)
    p=sub.add_parser('collect')
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        base.require(args.resamples>=1000,'Use at least 1000 bootstrap replicates')
        prepare(args.old_manifest,args.output_dir,args.resamples,args.seed)
    else:
        context_path=args.manifest.resolve().parent/'reader_context.json'
        context=base.read_json(context_path)
        base.verified(context['wrapper_code'])
        base.require(context['manifest']==base.reference(args.manifest),'Reader manifest differs')
        if args.command=='run':
            reader.run(context_path,args.case)
        else:
            base.verified(context['code'])
            fd.merge(args.manifest,plots=not args.no_plots)


if __name__=='__main__':
    main()
