"""Compare a supplied trained Lens with Logit Lens and the actual API readout."""

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from transformer_observatory.api import create_app
from transformer_observatory.lenses import apply_lens, register_lens
from transformer_observatory.models import load_model, load_tokenizer, register_model
from transformer_observatory.storage import Store, atomic_json

PROMPTS = [
    "A scientist placed a glass of water on the table. After several minutes, she measured its temperature and wrote the result in her notebook.",
    "The train arrived at the station before sunrise. A small group of travelers waited beside the platform with their bags and tickets.",
    "In a quiet library, a student opened a book about astronomy. The first chapter described how stars form inside clouds of gas and dust.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    store = Store(root / "data", root / "traces")
    registration = register_model(args.model)
    store.put("models", registration["id"], registration)
    lens = register_lens(store, registration, args.lens)
    app = create_app(store, testing=True)
    started = time.monotonic()
    with TestClient(app) as client:
        client.headers["x-observatory-token"] = client.get("/api/session").json()["token"]
        response = client.post(
            "/api/analysis/jobs",
            json=dict(
                model_id=registration["id"], prompt=PROMPTS[0], level=2, device=args.device, dtype="float32"
            ),
        )
        response.raise_for_status()
        trace_id = response.json()["id"]
        deadline = time.monotonic() + 300
        while True:
            job = client.get(f"/api/analysis/jobs/{trace_id}").json()
            if job["state"] in {"completed", "failed", "cancelled"}:
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Capture exceeded five minutes")
            time.sleep(0.1)
        if job["state"] != "completed":
            raise RuntimeError(str(job))
        manifest = store.manifest(trace_id)
        token = len(manifest["tokens"]) - 1
        index = registration["layers"] // 2
        stage = f"layer.{index - 1}.mlp"
        body = dict(stage=stage, token=token, step=0)
        baseline = client.post(f"/api/traces/{trace_id}/lens", json=body)
        baseline.raise_for_status()
        actual = client.post(f"/api/traces/{trace_id}/lens", json={**body, "lens_id": lens["id"]})
        actual.raise_for_status()
        preview = actual.json()
        model = load_model(registration, args.device, torch.float32)
        tokenizer = load_tokenizer(registration)
        norm = model.transformer.ln_f if registration["architecture"] == "gpt2" else model.model.norm
        rows = []
        max_api_error = 0.0
        with torch.inference_mode():
            for prompt_index, prompt in enumerate(PROMPTS):
                inputs = torch.tensor([tokenizer.encode(prompt)], device=args.device)
                output = model(input_ids=inputs, output_hidden_states=True, use_cache=False)
                teacher_log = output.logits.float().log_softmax(-1)
                teacher_prob = teacher_log.exp()
                for depth in range(registration["layers"]):
                    name = "embedding" if depth == 0 else f"layer.{depth - 1}.mlp"
                    hidden = output.hidden_states[depth]
                    translated, _ = apply_lens(store, lens, registration, name, hidden)
                    logit_log = model.lm_head(norm(hidden)).float().log_softmax(-1)
                    tuned_log = model.lm_head(norm(translated)).float().log_softmax(-1)
                    logit_kl = float((teacher_prob * (teacher_log - logit_log)).sum(-1).mean())
                    tuned_kl = float((teacher_prob * (teacher_log - tuned_log)).sum(-1).mean())
                    rows.append(
                        dict(
                            prompt=prompt_index,
                            tokens=inputs.shape[1],
                            stage=name,
                            logit_kl=logit_kl,
                            tuned_kl=tuned_kl,
                        )
                    )
                    if prompt_index == 0 and depth == index:
                        expected = tuned_log[0, token].exp()
                        for candidate in preview["candidates"]:
                            max_api_error = max(
                                max_api_error,
                                abs(candidate["probability"] - float(expected[candidate["id"]])),
                            )
        assert max_api_error < 1e-6, max_api_error
        logit_mean = float(np.mean([r["logit_kl"] for r in rows]))
        tuned_mean = float(np.mean([r["tuned_kl"] for r in rows]))
        assert tuned_mean < logit_mean, (logit_mean, tuned_mean)
        report = dict(
            all_passed=True,
            model_fingerprint=registration["fingerprint"],
            lens_fingerprint=lens["fingerprint"],
            lens_format=lens["format"],
            training_info=lens["training_info"],
            environment=dict(
                os=platform.platform(),
                torch=torch.__version__,
                device=args.device,
                dtype="float32",
                gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
            ),
            prompts=PROMPTS,
            evaluation="Three short original English prompts; smoke check only, not a corpus-wide benchmark. Mean is equal-weight over prompt/stage means.",
            stages_per_prompt=registration["layers"],
            mean_logit_kl=logit_mean,
            mean_tuned_kl=tuned_mean,
            api_max_probability_error=max_api_error,
            api_stage=stage,
            api_tuned_preview=preview,
            api_logit_preview=baseline.json(),
            trace_integrity=manifest["integrity"],
            seconds=time.monotonic() - started,
            rows=rows,
        )
        atomic_json(root / "report.json", report)
        print(
            f"Tuned Lens passed: KL {logit_mean:.6f} -> {tuned_mean:.6f}; API error {max_api_error:.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
