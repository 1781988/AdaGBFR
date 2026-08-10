#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import argparse, json
from adagbfr.knowledge import convert_gbfr_record

def locate_records(obj):
    if isinstance(obj, list): return obj
    if isinstance(obj, dict):
        for key in ("metrics","terms","data","records"):
            if isinstance(obj.get(key), list): return obj[key]
        if obj and all(isinstance(v, dict) for v in obj.values()): return list(obj.values())
    raise ValueError("Unsupported GBFR knowledge-base JSON structure")

def main():
    ap=argparse.ArgumentParser(description="Convert the official GBFR FMKG JSON to AdaGBFR normalized JSON.")
    ap.add_argument("--gbfr-root", required=True); ap.add_argument("--output", default="data/gbfr/fmkg.json"); args=ap.parse_args()
    src=Path(args.gbfr_root)/"knowledge_base"/"term_en_final.json"
    if not src.exists(): raise FileNotFoundError(f"Cannot find {src}")
    with src.open("r",encoding="utf-8") as f: records=locate_records(json.load(f))
    metrics=[convert_gbfr_record(r).to_dict() for r in records]; metrics=[m for m in metrics if m.get("name")]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as f: json.dump({"schema_version":"adagbfr-0.1","metrics":metrics},f,ensure_ascii=False,indent=2)
    derived=sum(bool(x.get("formulas")) for x in metrics)
    print(f"Wrote {len(metrics)} metrics ({derived} derived, {len(metrics)-derived} atomic) -> {out}")
if __name__=="__main__": main()
