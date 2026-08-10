#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,json
from adagbfr.config import load_config
from adagbfr.factory import build_pipeline

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/demo.yaml"); ap.add_argument("--case",default="dynamic_margin"); args=ap.parse_args()
    cfg=load_config(args.config); cases=[]
    with Path("data/demo_questions.jsonl").open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): cases.append(json.loads(line))
    case=next((x for x in cases if x.get("id")==args.case),None)
    if not case: raise KeyError(f"Unknown demo case: {args.case}")
    result=build_pipeline(cfg).run(case["query"],case["context"],plan_override=case["plan"])
    print(json.dumps(result.to_dict(),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
