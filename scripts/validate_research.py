"""Offline research checks against a completed, idle validate_model.py run."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformer_observatory.jobs import Jobs
from transformer_observatory.research import Research
from transformer_observatory.schema import StudyRequest
from transformer_observatory.storage import Store, atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-run", required=True, help="Idle validate_model.py output directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.trace_run).resolve()
    captured = json.loads((root / "report.json").read_text(encoding="utf-8"))
    ids = [row["id"] for row in captured["results"]]
    store = Store(root / "data", root / "traces")
    jobs = Jobs(store)
    research = Research(store, jobs)
    manifests = [store.manifest(i) for i in ids]
    originals = [digest(store.path(i) / "manifest.json") for i in ids]
    results = []

    def run(name, **values):
        started = time.monotonic()
        study = research.submit(StudyRequest(**values))
        while True:
            state = research.get(study["id"])
            if state["state"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        if state["state"] != "completed":
            raise RuntimeError(json.dumps(state, ensure_ascii=False))
        report = state["report"]
        result = report["result"]
        if values["kind"] == "intervention":
            assert result["restoration_max_logit_error"] == 0
            if name in {"self_patch", "zero_strength", "causal_prefix"}:
                assert result["logit_l2"] == 0, (name, result)
            else:
                assert result["logit_l2"] > 0, (name, result)
        results.append(dict(name=name, seconds=time.monotonic() - started, report=report))
        return result

    try:
        common = dict(kind="intervention", trace_ids=ids[:1])
        run("self_patch", kind="intervention", trace_ids=[ids[0]] * 2, stage="layer.0.mlp")
        run("cross_patch", kind="intervention", trace_ids=ids, stage="layer.0.mlp")
        run("final_block_patch", kind="intervention", trace_ids=ids, stage=manifests[0]["stages"][-1])
        run("head_ablation", **common, mode="head_ablation", layer=0, head=0)
        run("mlp_ablation", **common, mode="mlp_ablation", layer=0)
        run("zero_strength", **common, mode="head_ablation", strength=0)
        run("causal_prefix", **common, mode="head_ablation", readout_position=0)
        stats = run("statistics", kind="statistics", trace_ids=ids)
        means = []
        for m in manifests:
            record = next(r for r in m["tensors"] if r["kind"] == "metrics")
            means.append(float(np.mean(store.load(m["id"], record)[:, 2, 0], dtype=np.float64)))
        row = next(
            r
            for r in stats["rows"]
            if r["stage"] == manifests[0]["stages"][2] and r["metric"] == manifests[0]["metric_names"][0]
        )
        np.testing.assert_allclose(
            [row["mean"], row["std"]], [np.mean(means), np.std(means, ddof=1)], rtol=1e-12
        )
        first = run("tsne", kind="tsne", trace_ids=ids[:1], seed=42, perplexity=20)
        second = run("tsne_repeat", kind="tsne", trace_ids=ids[:1], seed=42, perplexity=20)
        np.testing.assert_array_equal(first["coordinates"], second["coordinates"])
        assert first["point_count"] == len(manifests[0]["tokens"]) * len(manifests[0]["stages"])
        assert originals == [digest(store.path(i) / "manifest.json") for i in ids]
        report = dict(
            commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            environment=results[0]["report"]["environment"],
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            all_passed=True,
            original_traces_unchanged=True,
            deterministic_tsne=True,
            tuned_lens=dict(
                status="not_run", reason="A model-matched trained checkpoint must be supplied separately"
            ),
            results=results,
        )
        atomic_json(Path(args.output).resolve(), report)
        print(json.dumps({"all_passed": True, "checks": len(results), "output": args.output}))
    finally:
        jobs.close()


if __name__ == "__main__":
    main()
