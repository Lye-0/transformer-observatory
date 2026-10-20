"""Explicit interventions and exploratory analyses, kept separate from measured traces."""

import csv
import gc
import io
import platform
import threading
import time
import traceback
import uuid

import numpy as np
import torch
from sklearn.manifold import TSNE

from .analysis import records
from .capture import Cancelled
from .models import load_model, load_tokenizer
from .schema import AnalysisRequest
from .storage import digest


def check_cancel(event):
    if event.is_set():
        raise Cancelled()


def prompt_tokens(manifest):
    return [t for t in manifest["tokens"] if not t["generated"]]


def verified_load(store, manifest, record):
    if digest(store.tensor_path(manifest["id"], record)) != record["sha256"]:
        raise ValueError("研究入力テンソルのハッシュが一致しません")
    return store.load(manifest["id"], record)


def residual_at(store, manifest, stage, position):
    record = next((r for r in records(manifest) if r["stage"] == stage and position in r["positions"]), None)
    if not record:
        raise ValueError("指定位置のResidualが未取得です。Level 2以上のトレースが必要です")
    return np.asarray(
        verified_load(store, manifest, record)[record["positions"].index(position)]
    ).copy(), record


def install_intervention(model, request, positions, vectors):
    """Return a removable hook. Patch only boundaries that replace the actual residual."""
    layers = model.transformer.h if model.config.model_type == "gpt2" else model.model.layers

    def alter(value):
        if request.strength == 0:
            return value
        result = value.clone()
        if request.mode == "residual_patch":
            replacement = torch.tensor(np.stack(vectors), device=value.device, dtype=value.dtype)
            if request.strength == 1:
                result[:, positions] = replacement
            else:
                result[:, positions] = (
                    value[:, positions] * (1 - request.strength) + replacement * request.strength
                )
        elif request.mode == "head_ablation":
            heads = (
                model.config.n_head if model.config.model_type == "gpt2" else model.config.num_attention_heads
            )
            if request.head >= heads:
                raise ValueError("Headが範囲外です")
            width = value.shape[-1] // heads
            result[:, positions, request.head * width : (request.head + 1) * width] *= 1 - request.strength
        else:
            result[:, positions] *= 1 - request.strength
        return result

    def before(module, args):
        return (alter(args[0]), *args[1:])

    def after(module, args, output):
        if isinstance(output, tuple):
            return (alter(output[0]), *output[1:])
        return alter(output)

    if request.mode == "residual_patch":
        if request.stage == "embedding":
            return layers[0].register_forward_pre_hook(before)
        pieces = request.stage.split(".")
        if len(pieces) != 3 or pieces[0] != "layer" or pieces[2] != "mlp" or not pieces[1].isdigit():
            raise ValueError(
                "Residual patchはEmbeddingまたはBlockのMLP後に対応します。Attention後は未対応です"
            )
        index = int(pieces[1])
        if index >= len(layers):
            raise ValueError("Layerが範囲外です")
        return layers[index].register_forward_hook(after)
    if request.layer >= len(layers):
        raise ValueError("Layerが範囲外です")
    block = layers[request.layer]
    if request.mode == "mlp_ablation":
        return block.mlp.register_forward_hook(after)
    projection = block.attn.c_proj if model.config.model_type == "gpt2" else block.self_attn.o_proj
    return projection.register_forward_pre_hook(before)


class Research:
    def __init__(self, store, jobs):
        self.store, self.jobs = store, jobs

    def update(self, study_id, **values):
        with self.store.lock:
            item = self.store.get("studies", study_id)
            item.update(values, updated_at=time.time())
            self.store.put("studies", study_id, item)

    def submit(self, request):
        with self.store.lock:
            return self._submit(request)

    def _submit(self, request):
        manifests = [self.store.manifest(i) for i in request.trace_ids]
        if any(m["status"] != "completed" for m in manifests):
            raise ValueError("完了トレースを選択してください")
        if len({m["model"]["fingerprint"] for m in manifests}) != 1:
            raise ValueError("同じ重み・Tokenizerのトレースだけを選択してください")
        if request.kind == "statistics" and len(manifests) < 2:
            raise ValueError("統計比較には2件以上の実験が必要です")
        if request.kind == "statistics" and len(set(request.trace_ids)) != len(request.trace_ids):
            raise ValueError("同じ実験を重複して統計比較できません")
        if request.kind == "tsne" and len(manifests) != 1:
            raise ValueError("追加射影は1実験を指定してください")
        if request.kind == "intervention" and request.mode == "residual_patch" and len(manifests) != 2:
            raise ValueError("Patchは介入先・取得元の2実験を指定してください（同じ実験も指定可能）")
        study_id = uuid.uuid4().hex
        self.jobs.cancellations[study_id] = threading.Event()
        item = dict(
            id=study_id,
            kind=request.kind,
            request=request.model_dump(),
            state="queued",
            phase="待機中",
            created_at=time.time(),
        )
        self.store.put("studies", study_id, item)
        self.jobs.executor.submit(self.run, study_id, request)
        return item

    def cancel(self, study_id):
        with self.store.lock:
            item = self.get(study_id)
            event = self.jobs.cancellations.get(study_id)
            if item["state"] not in {"completed", "failed", "cancelled"} and event:
                event.set()
                self.update(study_id, phase="安全な演算境界で中止します")
            return self.get(study_id)

    def get(self, study_id):
        item = self.store.get("studies", study_id)
        if not item:
            raise ValueError("研究ジョブが見つかりません")
        return item

    def run(self, study_id, request):
        event = self.jobs.cancellations[study_id]
        started = time.monotonic()
        try:
            check_cancel(event)
            self.update(study_id, state="running", phase="入力・記録条件を検証")
            manifests = [self.store.manifest(i) for i in request.trace_ids]

            def phase(message):
                self.update(study_id, phase=message)

            if request.kind == "intervention":
                result = self.intervene(request, manifests, event, phase)
            elif request.kind == "statistics":
                result = self.statistics(manifests, event)
            else:
                result = self.tsne(request, manifests[0], event, phase)
            check_cancel(event)
            report = dict(
                schema_version=1,
                kind=request.kind,
                request=request.model_dump(),
                model_fingerprint=manifests[0]["model"]["fingerprint"],
                elapsed_seconds=time.monotonic() - started,
                sources=[
                    dict(
                        id=m["id"],
                        title=m["title"],
                        dtype=m["dtype"],
                        device=m["device"],
                        created_at=m["created_at"],
                        integrity=m.get("integrity"),
                        manifest_sha256=digest(self.store.path(m["id"]) / "manifest.json"),
                    )
                    for m in manifests
                ],
                environment=dict(os=platform.platform(), torch=torch.__version__, numpy=np.__version__),
                result=result,
            )
            self.update(
                study_id, state="completed", phase="研究レポート確定", completed_at=time.time(), report=report
            )
        except Cancelled:
            self.update(study_id, state="cancelled", phase="中止しました")
        except Exception as exc:
            self.update(
                study_id,
                state="failed",
                phase="研究処理に失敗",
                error=str(exc),
                diagnostic=traceback.format_exc(),
            )
        finally:
            self.jobs.cancellations.pop(study_id, None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def intervene(self, request, manifests, event, phase):
        target = manifests[0]
        registration = next(
            (m for m in self.jobs.models() if m["fingerprint"] == target["model"]["fingerprint"]), None
        )
        if not registration:
            raise ValueError("同じ重み・TokenizerのモデルをこのOSに登録してください")
        tokens = prompt_tokens(target)
        positions = request.positions or [len(tokens) - 1]
        readout = request.readout_position if request.readout_position is not None else len(tokens) - 1
        if any(p >= len(tokens) for p in positions) or readout >= len(tokens):
            raise ValueError("介入先・読み出し位置には入力Prefill内の位置を指定してください")
        device = "cuda" if target["device"] == "cuda" and torch.cuda.is_available() else "cpu"
        dtype = target["dtype"] if device == "cuda" else "float32"
        self.jobs.preflight(
            AnalysisRequest(
                model_id=registration["id"], prompt=target["request"]["prompt"], device=device, dtype=dtype
            )
        )
        tokenizer = load_tokenizer(registration)
        if tokenizer.encode(target["request"]["prompt"]) != [t["id"] for t in tokens]:
            raise ValueError("入力トークンが元トレースと一致しません")
        vectors, source_records = [], []
        if request.mode == "residual_patch":
            source = manifests[1]
            src_positions = request.source_positions or [prompt_tokens(source)[-1]["position"]]
            if len(src_positions) != len(positions):
                raise ValueError("Patchの取得元と介入先の位置数を揃えてください")
            for p in src_positions:
                vector, r = residual_at(self.store, source, request.stage, p)
                vectors.append(vector)
                source_records.append(dict(position=p, file=r["file"], sha256=r["sha256"], step=r["step"]))
        phase("モデル読込・介入なしの基準を計算")
        model, handle = None, None
        cancellation_handles = []
        try:
            model = load_model(registration, device, getattr(torch, dtype))
            blocks = model.transformer.h if model.config.model_type == "gpt2" else model.model.layers
            for block in blocks:
                cancellation_handles.append(block.register_forward_pre_hook(lambda *_: check_cancel(event)))
            inputs = torch.tensor([[t["id"] for t in tokens]], device=device)

            def forward():
                check_cancel(event)
                with torch.inference_mode():
                    return model(input_ids=inputs, use_cache=False).logits[0, readout].float().cpu().double()

            baseline = forward()
            phase("指定位置への介入を適用して再計算")
            handle = install_intervention(model, request, positions, vectors)
            changed = forward()
            handle.remove()
            handle = None
            phase("介入解除後の復元を確認")
            restored = forward()
            restore_error = float((restored - baseline).abs().max())
            if not torch.allclose(restored, baseline, atol=1e-6, rtol=1e-6):
                raise ValueError("介入解除後の出力が基準と一致しません")
            p, q = baseline.softmax(-1), changed.softmax(-1)
            ids = set(p.topk(min(10, len(p))).indices.tolist()) | set(
                q.topk(min(10, len(q))).indices.tolist()
            )
            if request.token_id is not None:
                if request.token_id >= len(p):
                    raise ValueError("読み出すToken IDが語彙の範囲外です")
                ids.add(request.token_id)
            return dict(
                mode=request.mode,
                positions=positions,
                source_records=source_records,
                readout_position=readout,
                sequence="original prompt prefill only; generated tokens excluded",
                device=device,
                dtype=dtype,
                strength=request.strength,
                restoration_max_logit_error=restore_error,
                kl_baseline_to_intervened=max(
                    0.0, float((p * (baseline.log_softmax(-1) - changed.log_softmax(-1))).sum())
                ),
                total_variation=float((p - q).abs().sum() / 2),
                logit_l2=float(torch.linalg.vector_norm(changed - baseline)),
                candidates=[
                    dict(
                        id=i,
                        text=tokenizer.decode([i]),
                        raw=str(tokenizer.convert_ids_to_tokens(i)),
                        baseline=float(p[i]),
                        intervened=float(q[i]),
                        difference=float(q[i] - p[i]),
                    )
                    for i in sorted(ids, key=lambda i: float(q[i]), reverse=True)
                ],
                interpretation="この位置・この介入・この入力条件での出力差。一般的な重要性や自然な因果機構を保証しない。",
            )
        finally:
            if handle:
                handle.remove()
            for h in cancellation_handles:
                h.remove()
            model = None

    def statistics(self, manifests, event):
        reference = manifests[0]
        if any(
            m["stages"] != reference["stages"] or m["metric_names"] != reference["metric_names"]
            for m in manifests
        ):
            raise ValueError("観測点と指標の定義を揃えてください")
        groups = {}
        per_experiment = []
        for m in manifests:
            check_cancel(event)
            metric = next(r for r in m["tensors"] if r["kind"] == "metrics")
            mask = next((r for r in m["tensors"] if r["kind"] == "metric_validity"), None)
            values = verified_load(self.store, m, metric)
            validity = verified_load(self.store, m, mask) if mask else np.ones(values.shape, dtype=bool)
            for s, stage in enumerate(m["stages"]):
                for k, name in enumerate(m["metric_names"]):
                    if name == "projected_update":
                        continue  # Independent PCAs cannot be pooled.
                    valid = values[:, s, k][validity[:, s, k].astype(bool)]
                    if not len(valid):
                        continue
                    mean = float(np.mean(valid, dtype=np.float64))
                    row = dict(trace_id=m["id"], stage=stage, metric=name, mean=mean, token_count=len(valid))
                    per_experiment.append(row)
                    groups.setdefault((stage, name), []).append(mean)
        return dict(
            unit="各実験内で有効トークンを平均し、その実験平均を同じ重みで集計",
            excluded=["projected_update: 各実験の独立PCAに依存するため除外"],
            interpretation="記述統計。トークンを独立した実験反復とは扱わず、p値・因果性を主張しない。入力長や生成条件の違いにも注意。",
            rows=[
                dict(
                    stage=s,
                    metric=k,
                    experiments=len(v),
                    mean=float(np.mean(v)),
                    std=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    min=min(v),
                    max=max(v),
                )
                for (s, k), v in groups.items()
            ],
            per_experiment=per_experiment,
        )

    def tsne(self, request, m, event, phase):
        if not records(m):
            raise ValueError("追加射影にはLevel 2以上のResidualが必要です")
        count, stages, width = len(m["tokens"]), len(m["stages"]), m["model"]["hidden_size"]
        total = count * stages
        estimate = total * width * 4 * 6 + total**2 * 8
        if total > 4096 or estimate > self.store.settings["ram_budget_mib"] * 2**20:
            raise ValueError(
                "全点t-SNEの上限4096観測点またはRAM予算を超えます。一部抽出せず、短い入力で解析してください"
            )
        if request.perplexity >= total:
            raise ValueError("Perplexityは観測点数より小さくしてください")
        values = np.empty((total, width), dtype=np.float32)
        for r in records(m):
            check_cancel(event)
            s = m["stages"].index(r["stage"])
            values[np.array(r["positions"]) * stages + s] = verified_load(self.store, m, r)
        phase(f"全{total}点のt-SNEを計算中（中止はこの計算区間の終了時に反映）")
        estimator = TSNE(
            n_components=3,
            perplexity=request.perplexity,
            random_state=request.seed,
            init="pca",
            learning_rate="auto",
            max_iter=750,
            n_jobs=1,
        )
        coordinates = estimator.fit_transform(values)
        if not np.isfinite(coordinates).all():
            raise ValueError("t-SNEに非有限値が発生しました")
        return dict(
            algorithm="t-SNE / Barnes-Hut",
            coordinates=coordinates.reshape(count, stages, 3).tolist(),
            stages=m["stages"],
            tokens=m["tokens"],
            seed=request.seed,
            perplexity=request.perplexity,
            iterations=int(estimator.n_iter_),
            kl_divergence=float(estimator.kl_divergence_),
            point_count=total,
            interpretation="近傍の探索用の非線形射影。実測の意味空間・高次元距離・変位量を再現しない。Scientific/DisplacementのPCAは変更しない。",
        )


def report_csv(report):
    result = report["result"]
    if report["kind"] == "intervention":
        rows = result["candidates"]
    elif report["kind"] == "statistics":
        rows = result["rows"]
    else:
        rows = [
            dict(position=t, token_id=token["id"], stage=name, x=xyz[0], y=xyz[1], z=xyz[2])
            for t, token in enumerate(result["tokens"])
            for name, xyz in zip(result["stages"], result["coordinates"][t])
        ]
    output = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            safe = {
                key: "'" + value
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r"))
                else value
                for key, value in row.items()
            }
            writer.writerow(safe)
    return output.getvalue()
