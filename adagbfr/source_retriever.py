from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_%.-]*|[\u4e00-\u9fff]")

def tokenize(text: str) -> List[str]:
    return [x.lower() for x in _TOKEN_RE.findall(text or "")]

@dataclass
class SourceDoc:
    source_id: str
    title: str
    text: str
    authority: float = 0.8
    url: str = ""

    @classmethod
    def from_dict(cls, obj: Dict) -> "SourceDoc":
        return cls(
            source_id=str(obj.get("source_id") or obj.get("id") or obj.get("title") or "source"),
            title=str(obj.get("title") or ""),
            text=str(obj.get("text") or obj.get("content") or ""),
            authority=float(obj.get("authority", 0.8)),
            url=str(obj.get("url") or ""),
        )

@dataclass
class RetrievedSource:
    doc: SourceDoc
    score: float

class LocalSourceIndex:
    """Deterministic BM25 index for query-time metric expansion sources."""
    def __init__(self, docs: Iterable[SourceDoc]):
        self.docs = list(docs)
        self.tokens = [tokenize(d.title + " " + d.text) for d in self.docs]
        self.tfs = [Counter(t) for t in self.tokens]
        self.avgdl = sum(len(t) for t in self.tokens) / max(1, len(self.tokens))
        self.df: Counter[str] = Counter()
        for toks in self.tokens:
            self.df.update(set(toks))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LocalSourceIndex":
        docs: List[SourceDoc] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(SourceDoc.from_dict(json.loads(line)))
        return cls(docs)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedSource]:
        q = tokenize(query)
        if not q or not self.docs:
            return []
        k1, b, n = 1.5, 0.75, len(self.docs)
        scored: List[RetrievedSource] = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            dl = max(1, len(self.tokens[i])); tf = self.tfs[i]
            for term in q:
                freq = tf.get(term, 0)
                if not freq: continue
                df = self.df.get(term, 0)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1 - b + b * dl / max(1.0, self.avgdl))
                score += idf * (freq * (k1 + 1) / denom)
            score *= max(0.1, min(1.0, doc.authority))
            if score > 0:
                scored.append(RetrievedSource(doc=doc, score=score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
