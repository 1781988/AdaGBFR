#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,json,random
from tqdm import tqdm
from adagbfr.config import load_config,resolve_path
from adagbfr.datasets import load_dataset
from adagbfr.evaluation import score_prediction,summarize
from adagbfr.factory import build_pipeline
from adagbfr.knowledge import MetricStore

def main():
    ap=argparse.ArgumentParser(description="Run AdaGBFR on one supported financial benchmark.")
    ap.add_argument("--config",default="configs/adagbfr.yaml"); ap.add_argument("--dataset",required=True,choices=["FinQA","TAT-QA","FinanceReasoning","CFBenchmark"])
    ap.add_argument("--data-root",required=True); ap.add_argument("--limit",type=int,default=0); ap.add_argument("--seed",type=int,default=42); ap.add_argument("--output",required=True); args=ap.parse_args()
    cfg=load_config(args.config); base_store=MetricStore.from_json(resolve_path(cfg,cfg["knowledge_base"]["path"])); rows=load_dataset(args.dataset,args.data_root)
    rng=random.Random(args.seed); rng.shuffle(rows)
    if args.limit>0: rows=rows[:args.limit]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); records=[]
    with out.open("w",encoding="utf-8") as f:
        for i,row in enumerate(tqdm(rows,desc=args.dataset)):
            result=build_pipeline(cfg,base_store=base_store).run(row["query"],row["context"]); result_dict=result.to_dict(); score=score_prediction(result_dict,row["answer"])
            record={"id":i,"query":row["query"],"ground_truth":row["answer"],"source_file":row["source_file"],"result":result_dict,"score":score}
            records.append(record); f.write(json.dumps(record,ensure_ascii=False)+"\n"); f.flush()
    summary=summarize(records); summary["dataset"]=args.dataset; summary["config"]=args.config
    summary_path=out.with_suffix(out.suffix+".summary.json")
    with summary_path.open("w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print(f"Detailed results: {out}"); print(f"Summary: {summary_path}")
    print("Built-in scoring is for fast iteration. Use each benchmark's official evaluation script for paper tables.")
if __name__=="__main__": main()
