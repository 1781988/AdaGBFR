#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,json,random

def main():
    ap=argparse.ArgumentParser(description="Mask a fraction of derived FMKG nodes to test dynamic recovery.")
    ap.add_argument("--kb",required=True); ap.add_argument("--ratio",type=float,required=True); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--output",required=True); ap.add_argument("--manifest",default=None); args=ap.parse_args()
    if not 0<args.ratio<1: raise ValueError("ratio must be between 0 and 1")
    with Path(args.kb).open("r",encoding="utf-8") as f: obj=json.load(f)
    metrics=obj.get("metrics",obj); derived=[m for m in metrics if m.get("formulas")]; rng=random.Random(args.seed)
    mask_n=max(1,round(len(derived)*args.ratio)); masked_names={m["name"] for m in rng.sample(derived,mask_n)}
    kept=[m for m in metrics if m.get("name") not in masked_names]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f: json.dump({"schema_version":"adagbfr-0.1","metrics":kept},f,ensure_ascii=False,indent=2)
    manifest=Path(args.manifest or (str(out)+".masked.json"))
    with manifest.open("w",encoding="utf-8") as f: json.dump({"ratio":args.ratio,"seed":args.seed,"masked_metrics":sorted(masked_names)},f,ensure_ascii=False,indent=2)
    print(f"Masked {len(masked_names)}/{len(derived)} derived metrics -> {out}"); print(f"Manifest -> {manifest}")
if __name__=="__main__": main()
