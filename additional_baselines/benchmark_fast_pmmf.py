"""Read a Pet prefix without loading the full 2.5GB artifact; benchmark only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import resource

from .candidates import ExperimentInputs, Request, file_reference
from .rerank import FairDiverseReranker
from .rerank.fast_pmmf import FastPMMFReranker


def load_prefix(path,count):
    header = []
    with Path(path).open() as stream:
        for line in stream:
            if line.strip() == '"requests": [':
                break
            header.append(line)
        else:
            raise ValueError("Expected the candidate writer's requests array")
        data = json.loads(''.join(header)+'"requests": []}')
        decoder, buffer, requests = json.JSONDecoder(), "", []
        while len(requests)<count:
            buffer = buffer.lstrip(" \r\n\t,")
            if buffer.startswith("]"):
                break
            try:
                value,end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(65536)
                if not chunk:
                    raise ValueError("Incomplete request prefix")
                buffer += chunk
                continue
            requests.append(Request(**value))
            buffer = buffer[end:]
    data["item_groups"] = {name:{int(i):g for i,g in mapping.items()} for name,mapping in data["item_groups"].items()}
    data["requests"] = requests
    return ExperimentInputs(**data).validate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--legacy-requests",type=int,default=3)
    parser.add_argument("--fast-requests",type=int,default=2000)
    args = parser.parse_args()
    inputs = load_prefix(args.candidates,args.fast_requests)
    small = ExperimentInputs(deepcopy(inputs.metadata),inputs.item_groups,inputs.requests[:args.legacy_requests])
    weights = dict.fromkeys(set(inputs.item_groups["brand"].values()),1.)
    trials = []
    for strength in (0.,1.,10.,30.):
        options = dict(model="PMMF",group="brand",fairness_lambda=strength,weights=weights)
        start = time.perf_counter()
        legacy = FairDiverseReranker(**options).rerank(small,5)
        legacy_seconds = time.perf_counter()-start
        start = time.perf_counter()
        exact_small = FastPMMFReranker(**options).rerank(small,5)
        exact_small_seconds = time.perf_counter()-start
        equal = legacy.rankings == exact_small.rankings
        if not equal:
            raise ValueError(f"Pet prefix ranking disagreement at lambda={strength}")
        start = time.perf_counter()
        fast = FastPMMFReranker(**options).rerank(inputs,5)
        fast_seconds = time.perf_counter()-start
        trials.append(dict(strength=strength,legacy_requests=len(small.requests),legacy_seconds=legacy_seconds,
            parity_rankings_equal=equal,fast_parity_seconds=exact_small_seconds,fast_requests=len(inputs.requests),
            fast_seconds=fast_seconds,projected_full_594800_seconds=fast_seconds/len(inputs.requests)*594800))
        print(json.dumps(trials[-1]),flush=True)
        del fast
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(scope="benchmark_prefix; not validation selection",trials=trials,
        catalog_size=len(inputs.item_groups["brand"]),brand_groups=len(weights),
        candidate_source=file_reference(args.candidates),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2),indent=2)+"\n")


if __name__ == "__main__":
    main()
