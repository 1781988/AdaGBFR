#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,json

def main():
    ap=argparse.ArgumentParser(description="Build a retrieval corpus from a normalized FMKG for coverage-stress experiments.")
    ap.add_argument("--kb",required=True); ap.add_argument("--output",required=True); ap.add_argument("--authority",type=float,default=0.9); args=ap.parse_args()
    with Path(args.kb).open("r",encoding="utf-8") as f: obj=json.load(f)
    metrics=obj.get("metrics",obj); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8") as w:
        for i,m in enumerate(metrics):
            formulas=[x.get("expression","") for x in m.get("formulas",[]) if x.get("expression")]
            if not formulas: continue
            text=(m.get("definition") or "")+"\n"+"\n".join(f"Formula: {x}" for x in formulas)
            w.write(json.dumps({"source_id":f"fmkg_doc_{i}","title":m.get("name",""),"text":text,"authority":args.authority,"url":""},ensure_ascii=False)+"\n")
    print(f"Wrote retrieval corpus -> {out}")
    print("NOTE: controlled stress-test corpus only; use independent sources in the main paper to avoid leakage.")
if __name__=="__main__": main()
