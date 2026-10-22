import gc
import platform
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import psutil
import torch
import transformers

from .analysis import project_trace
from .capture import Cancelled, Capture
from .model_discovery import inspect_model
from .models import TINY_ID, capabilities, load_model, load_tokenizer, tiny_registration, tokenize
from .runtime import configure_runtime
from .schema import TERMINAL, AnalysisRequest
from .storage import TraceWriter


class Jobs:
    def __init__(self, store):
        configure_runtime()
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="observatory")
        self.cancellations = {}
        self.closed = False

    def models(self):
        models = [self.current_capabilities(m) for m in [tiny_registration(), *self.store.list("models")]]
        for model in models:
            inspection = inspect_model(model["path"]) if model.get("path") else None
            model["available"] = inspection is None or inspection["status"] == "available"
            model["availability_reason"] = inspection["reason"] if inspection else "内蔵の数値検証モデル"
        return models

    def current_capabilities(self, model):
        current = capabilities(model["architecture"])
        for key in ["interventions", "tuned_lens"]:
            model["capabilities"][key] = current[key]
        if any(lens["model_fingerprint"] == model["fingerprint"] for lens in self.store.list("lenses")):
            model["capabilities"]["tuned_lens"] = "学習済みローカル重みを登録済み"
        return model

    def model(self, model_id):
        if model_id == TINY_ID:
            return self.current_capabilities(tiny_registration())
        registration = self.store.get("models", model_id)
        if not registration:
            raise ValueError("モデルが登録されていません")
        inspection = inspect_model(registration["path"])
        if inspection["status"] != "available":
            raise ValueError(f"Model Managerで保存場所を再指定してください: {inspection['reason']}")
        return self.current_capabilities(registration)

    def preflight(self, request):
        m = self.model(request.model_id)
        tokens = tokenize(load_tokenizer(m), request.prompt)
        length = len(tokens) + (request.max_new_tokens if request.generate else 0)
        if length > min(request.max_tokens, m["max_tokens"]):
            raise ValueError(
                f"入力＋生成の上限を超えます: {length} / {min(request.max_tokens, m['max_tokens'])} tokens"
            )
        if any(i >= m["layers"] for i in request.layers) or any(i >= m["heads"] for i in request.heads):
            raise ValueError("指定Layer/Headがモデルの範囲外です")
        if request.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDAを利用できません")
        stages = 2 * m["layers"] + 1
        residual = stages * length * m["hidden_size"] * 4  # float32 archive preserves BF16 values exactly
        selected_layers = len(request.layers) or m["layers"]
        selected_heads = len(request.heads) or m["heads"]
        # Conservative provisional estimate, clearly distinguished from measured file sizes.
        detail = selected_layers * length * m["hidden_size"] * 4 * 24 if request.level >= 3 else 0
        attn = (
            selected_layers * selected_heads * length**2 * 4
            if request.level >= 3 and request.attention
            else 0
        )
        spool = residual + detail + attn
        disk_free = shutil.disk_usage(self.store.trace_root).free
        if spool > self.store.settings["trace_limit_gib"] * 2**30 or spool + 64 * 2**20 > disk_free:
            raise ValueError(
                "推定トレース容量が保存上限または空き容量を超えます。範囲・長さを減らしてください"
            )
        device = (
            "cuda"
            if request.device == "cuda" or (request.device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        dtype = (
            request.dtype
            if request.dtype != "auto"
            else ("float32" if device == "cpu" or m["id"] == TINY_ID else "bfloat16")
        )
        # Eager attention materializes scores/probabilities; include them even without archival.
        element_bytes = 4 if dtype == "float32" else 2
        weight_memory = m.get("weight_bytes", 0) * (2 if dtype == "float32" else 1)
        eager_workspace = m["heads"] * length**2 * (element_bytes * 2 + 4)
        kv_memory = 2 * m["layers"] * length * m["hidden_size"] * element_bytes
        activation_memory = length * m["hidden_size"] * 48 * element_bytes
        device_estimate = weight_memory + eager_workspace + kv_memory + activation_memory + 256 * 2**20
        host_estimate = weight_memory + (
            eager_workspace * 2 + activation_memory * 3
            if request.level >= 3 or device == "cpu"
            else activation_memory
        )
        host_estimate += self.store.settings["ram_budget_mib"] * 2**20
        ram_available = psutil.virtual_memory().available
        gpu_free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
        if host_estimate > ram_available * 0.85:
            raise ValueError("推定RAM使用量が空き容量に近すぎます。入力長・取得範囲を減らしてください")
        if device == "cuda" and device_estimate > gpu_free * 0.9:
            raise ValueError("推定VRAM使用量が空き容量に近すぎます。入力長を減らすかCPUを選んでください")
        if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("このGPUはbfloat16をサポートしていません。float32またはfloat16を選んでください")
        warnings = []
        warnings.append(
            "RAM・VRAMはEager AttentionとKV Cacheを含む概算で、ピーク使用量の保証ではありません。"
        )
        if request.level >= 3:
            warnings.append("詳細容量は概算です。実行時も容量上限を監視します。")
        if request.level == 1:
            warnings.append("座標確定まではResidualを一時保存します。完了後は高次元値を破棄します。")
        if device == "cpu" and m.get("weight_bytes", 0) > 2**30:
            warnings.append("CPUで実モデルを解析します。処理時間が長くなる場合があります。")
        return dict(
            model=m,
            tokens=tokens,
            input_tokens=len(tokens),
            max_total_tokens=length,
            stages=stages,
            trajectory_count=len(tokens),
            device=device,
            dtype=dtype,
            estimated_spool_bytes=spool,
            estimated_final_bytes=(length * stages * 40 if request.level == 1 else spool),
            ram_available_bytes=ram_available,
            estimated_ram_bytes=host_estimate,
            estimated_vram_bytes=device_estimate if device == "cuda" else 0,
            disk_free_bytes=disk_free,
            gpu_free_bytes=gpu_free,
            weight_bytes=m.get("weight_bytes", 0),
            warnings=warnings,
        )

    def submit(self, request):
        self.preflight(request)
        trace_id = uuid.uuid4().hex
        self.cancellations[trace_id] = threading.Event()
        self.update(
            trace_id,
            state="queued",
            stage=0,
            percent=0,
            detail="待機中",
            request=request.model_dump(),
            created_at=time.time(),
            trace_id=trace_id,
        )
        self.executor.submit(self.run, trace_id, request)
        return self.store.get("jobs", trace_id)

    def update(self, job_id, **values):
        with self.store.lock:
            previous = self.store.get("jobs", job_id) or dict(id=job_id)
            previous.update(values, updated_at=time.time())
            self.store.put("jobs", job_id, previous)

    def cancel(self, trace_id):
        j = self.store.get("jobs", trace_id)
        if not j:
            raise ValueError("ジョブがありません")
        if j["state"] in TERMINAL:
            return j
        self.cancellations[trace_id].set()
        self.update(trace_id, state="cancelling", detail="安全な演算境界で中止します")
        return self.store.get("jobs", trace_id)

    def run(self, trace_id, request: AnalysisRequest):
        model, capture, writer = None, None, None
        event = self.cancellations[trace_id]
        try:
            if event.is_set():
                raise Cancelled()
            self.update(trace_id, state="preflight", stage=0, percent=2, detail="リソースと取得範囲を確認")
            pf = self.preflight(request)
            registration = pf["model"]
            manifest = dict(
                title=request.title or request.prompt[:48],
                level=request.level,
                model=registration,
                request=request.model_dump(),
                device=pf["device"],
                dtype=pf["dtype"],
                tokens=pf["tokens"],
                generation=[],
                checks=[],
                environment=dict(
                    os=platform.platform(),
                    torch=torch.__version__,
                    transformers=transformers.__version__,
                    numpy=np.__version__,
                    python=platform.python_version(),
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    matmul_precision=dict(
                        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                        tf32=torch.backends.cuda.matmul.allow_tf32,
                    ),
                ),
                capture_dtype="float32",
                capture_backend="pytorch-eager",
                tensor_provenance="hook=actual module input/output; reconstructed=measured values + weights with validated adapter math; backend_mask=mask converted to boolean",
                generation_backend="full-context recompute" if registration["id"] == TINY_ID else "KV cache",
                observation_definition="embedding + attention_residual + mlp_residual; final norm excluded",
            )
            kv_heads = registration.get("kv_heads") or registration["heads"]
            manifest["query_to_kv_heads"] = [
                h // (registration["heads"] // kv_heads) for h in range(registration["heads"])
            ]
            writer = TraceWriter(self.store, trace_id, manifest)
            self.update(trace_id, state="loading", stage=1, percent=7, detail="モデルと解析アダプターを準備")
            model = load_model(registration, pf["device"], getattr(torch, pf["dtype"]))
            if event.is_set():
                raise Cancelled()
            self.update(
                trace_id, state="tokenizing", stage=2, percent=18, detail=f"{pf['input_tokens']} tokens"
            )
            tokenizer = load_tokenizer(registration)
            tokens = manifest["tokens"]
            ids = [t["id"] for t in tokens]

            def progress(layer, count, step):
                total = 1 + (request.max_new_tokens if request.generate else 0)
                pct = 22 + 55 * (step + layer / count) / total
                self.update(
                    trace_id,
                    state="capturing",
                    stage=3,
                    percent=round(pct, 1),
                    detail=f"Layer {layer + 1} / {count} · " + ("Prefill" if step == 0 else f"Decode {step}"),
                )

            capture = Capture(model, writer, request, event, progress).install()
            generator = torch.Generator(device=pf["device"]).manual_seed(request.seed)
            cache = None
            input_tensor = torch.tensor([ids], dtype=torch.long, device=pf["device"])
            generated = []
            # Process every predicted token before assigning it an internal trajectory.
            # The final generated token receives an explicit extra forward pass, including EOS.
            for step in range((request.max_new_tokens if request.generate else 0) + 1):
                if event.is_set():
                    raise Cancelled()
                recomputed = step > 0 and registration["id"] == TINY_ID
                capture.set_step(step, range(len(ids)) if step == 0 else [len(ids) - 1], recomputed)
                with torch.inference_mode():
                    result = model(
                        input_ids=input_tensor, past_key_values=cache, use_cache=registration["id"] != TINY_ID
                    )
                cache = result.past_key_values
                tokens[-1]["processed"] = True
                if step > 0:
                    manifest["generation"][-1]["processed_at_step"] = step
                if not request.generate or len(generated) >= request.max_new_tokens:
                    break
                if generated and generated[-1] == tokenizer.eos_token_id:
                    break
                logits = result.logits[0, -1].float()
                if request.temperature == 0:
                    token_id = int(logits.argmax())
                else:
                    probs = (logits / request.temperature).softmax(-1)
                    if request.top_p < 1:
                        sorted_p, order = probs.sort(descending=True)
                        mask = sorted_p.cumsum(0) - sorted_p > request.top_p
                        sorted_p[mask] = 0
                        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
                        probs /= probs.sum()
                    token_id = int(torch.multinomial(probs, 1, generator=generator))
                ids.append(token_id)
                generated.append(token_id)
                tokens.append(
                    dict(
                        position=len(ids) - 1,
                        id=token_id,
                        text=tokenizer.decode([token_id]),
                        raw=str(tokenizer.convert_ids_to_tokens(token_id)),
                        generated=True,
                        step=step + 1,
                        processed=False,
                    )
                )
                manifest["generation"].append(
                    dict(
                        token_id=token_id,
                        position=len(ids) - 1,
                        predicted_at_step=step,
                        processed_at_step=None,
                    )
                )
                writer.store.save_manifest(manifest)
                input_tensor = torch.tensor(
                    [ids if registration["id"] == TINY_ID else [token_id]],
                    dtype=torch.long,
                    device=pf["device"],
                )
            manifest["checks"] = capture.checks
            manifest["generated_text"] = tokenizer.decode(generated, skip_special_tokens=True)
            capture.close()
            self.update(
                trace_id,
                state="projecting",
                stage=4,
                percent=83,
                detail="全観測点の共通PCAと高次元指標を計算",
            )
            project_trace(self.store, writer, event)
            if event.is_set():
                raise Cancelled()
            self.update(trace_id, state="preparing", stage=5, percent=98, detail="整合性と描画データを確定")
            checks = manifest["checks"]
            writer.finalize(
                dict(
                    status="verified",
                    checks=len(checks),
                    max_abs_error=max((c["max_abs_error"] for c in checks), default=0),
                    max_scaled_error=max((c["max_scaled_error"] for c in checks), default=0),
                    all_passed=all(c["passed"] for c in checks),
                )
            )
            if registration["id"] != TINY_ID:
                r = self.store.get("models", registration["id"])
                r["last_validation"] = dict(
                    trace_id=trace_id,
                    time=time.time(),
                    level=request.level,
                    device=pf["device"],
                    dtype=pf["dtype"],
                )
                r["capabilities"]["residual"] = "検証済み"
                if request.level >= 3 and len(capture.selected) == registration["layers"]:
                    r["capabilities"]["attention"] = r["capabilities"]["mlp"] = "検証済み"
                if request.generate:
                    r["capabilities"]["generation"] = "検証済み"
                self.store.put("models", r["id"], r)
            self.update(trace_id, state="completed", stage=5, percent=100, detail="解析完了")
        except Cancelled:
            self.persist_checks(writer, capture)
            self.store.mark_incomplete(trace_id, "ユーザーによる中止")
            self.update(trace_id, state="cancelled", detail="中止 · 部分データを保持")
        except Exception as exc:
            self.persist_checks(writer, capture)
            reason = str(exc) or type(exc).__name__
            self.store.mark_incomplete(trace_id, reason)
            self.update(
                trace_id,
                state="failed",
                detail="解析に失敗しました",
                error=reason,
                diagnostic=traceback.format_exc(),
            )
        finally:
            if capture:
                capture.close()
            del model, capture
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.cancellations.pop(trace_id, None)

    def persist_checks(self, writer, capture):
        if writer and capture:
            writer.manifest["checks"] = capture.checks
            try:
                self.store.save_manifest(writer.manifest)
            except (OSError, ValueError):
                pass  # Keep the last atomically committed journal if the disk itself failed.

    def close(self):
        self.closed = True
        for event in list(self.cancellations.values()):
            event.set()
        self.executor.shutdown(wait=True, cancel_futures=False)

    def lens(self, trace_id, stage, token, step, lens_id=None):
        m = self.store.manifest(trace_id)
        r = next(
            (
                r
                for r in m["tensors"]
                if r["kind"] == "residual"
                and r["stage"] == stage
                and r["step"] == step
                and token in r["positions"]
            ),
            None,
        )
        if not r:
            raise ValueError("この位置のResidualは未取得です")
        model = None
        try:
            registration = next(
                (r for r in self.models() if r["fingerprint"] == m["model"]["fingerprint"]), None
            )
            if not registration:
                raise ValueError("同じ重み・TokenizerのモデルをこのOSのModel Managerに登録してください")
            device = "cuda" if m["device"] == "cuda" and torch.cuda.is_available() else "cpu"
            dtype = getattr(torch, m["dtype"]) if device == "cuda" else torch.float32
            model = load_model(registration, device, dtype)
            tokenizer = load_tokenizer(registration)
            x = np.asarray(self.store.load(trace_id, r)[r["positions"].index(token)]).copy()
            x = torch.tensor(x, device=device, dtype=dtype)
            lens_metadata = None
            if lens_id:
                from .lenses import apply_lens

                lens_registration = self.store.get("lenses", lens_id)
                if not lens_registration:
                    raise ValueError("学習済みLensが登録されていません")
                x, lens_metadata = apply_lens(self.store, lens_registration, registration, stage, x)
            norm = model.transformer.ln_f if model.config.model_type == "gpt2" else model.model.norm
            with torch.inference_mode():
                probabilities = model.lm_head(norm(x)).float().softmax(-1)
                values, indices = probabilities.topk(10)
            return dict(
                method="Tuned Lens (trained residual affine + final norm + LM head)"
                if lens_id
                else "Logit Lens (final norm + LM head)",
                lens=lens_metadata,
                device=device,
                dtype=str(dtype),
                candidates=[
                    dict(
                        id=int(i),
                        text=tokenizer.decode([int(i)]),
                        raw=str(tokenizer.convert_ids_to_tokens(int(i))),
                        probability=float(p),
                    )
                    for p, i in zip(values.cpu(), indices.cpu())
                ],
            )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
