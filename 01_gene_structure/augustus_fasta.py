#!/usr/bin/env python3
"""Write test-split sequences from a BEND-13 JSONL to FASTA (plus record metadata) for AUGUSTUS."""
import argparse, gzip, json, textwrap
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--jsonl', required=True)
ap.add_argument('--fasta', required=True)
ap.add_argument('--meta', required=True)
args = ap.parse_args()

Path(args.fasta).parent.mkdir(parents=True, exist_ok=True)
meta = []
with gzip.open(args.jsonl, 'rt') as inp, open(args.fasta, 'w') as fa:
    for line in inp:
        rec = json.loads(line)
        rid = rec['id']
        seq = rec['sequence'].upper()
        fa.write(f'>{rid}\n')
        fa.write('\n'.join(textwrap.wrap(seq, 80)) + '\n')
        meta.append({k: rec.get(k) for k in ['id','split','chrom','start0','end0','strand','gene_id','transcript_id','gene_name']})
with open(args.meta, 'w') as out:
    json.dump(meta, out, indent=2)
print(json.dumps({'records': len(meta), 'fasta': args.fasta, 'meta': args.meta}, indent=2))
