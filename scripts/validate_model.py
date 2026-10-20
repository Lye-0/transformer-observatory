"""Reproducible offline GPU smoke/consistency run; never downloads a model."""

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from transformer_observatory.analysis import compare
from transformer_observatory.jobs import Jobs
from transformer_observatory.models import register_model
from transformer_observatory.schema import AnalysisRequest
from transformer_observatory.storage import Store, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    store = Store(out / "data", out / "traces")
    registration = register_model(args.model, "Qwen3-0.6B")
    store.put("models", registration["id"], registration)
    jobs = Jobs(store)
    results = []
    started = time.monotonic()
    try:
        for text in ["The sky is blue.", "The sky is green."]:
            t = time.monotonic()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rss_peak = psutil.Process().memory_info().rss
            job = jobs.submit(
                AnalysisRequest(
                    model_id=registration["id"],
                    prompt=text,
                    title=text,
                    device=args.device,
                    level=4,
                    attention=True,
                    generate=True,
                    max_new_tokens=2,
                )
            )
            while True:
                rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
                state = store.get("jobs", job["id"])
                if state["state"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.1)
            if state["state"] != "completed":
                raise RuntimeError(json.dumps(state, ensure_ascii=False))
            m = store.manifest(job["id"])
            results.append(
                dict(
                    id=job["id"],
                    seconds=round(time.monotonic() - t, 3),
                    tokens=len(m["tokens"]),
                    checkpoints=len(m["stages"]),
                    integrity=m["integrity"],
                    dtype=m["dtype"],
                    device=m["device"],
                    fingerprint=m["model"]["fingerprint"],
                    size_bytes=m["size_bytes"],
                    sampled_peak_rss_bytes=rss_peak,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
                    if args.device == "cuda"
                    else 0,
                    peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else 0,
                    environment=m["environment"],
                )
            )
        comparison = compare(store, results[0]["id"], results[1]["id"])
        report = dict(
            results=results,
            elapsed=round(time.monotonic() - started, 3),
            common_prefix_tokens=comparison["common_prefix_tokens"],
            common_prefix_max_distance=comparison["common_prefix_max_distance"],
        )
        atomic_json(out / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
