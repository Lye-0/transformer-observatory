"""Offline adapter for AlignmentResearch linear Tuned Lens checkpoints."""

import hashlib
import json
import zipfile
from pathlib import Path

import torch
from safetensors.torch import load_file

from .storage import digest


def read_checkpoint(path, model, budget_mib):
    root = Path(path).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError("Lensのconfig.jsonがないか、大きすぎます")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("lens_type") != "linear_tuned_lens"
        or config.get("d_model") != model["hidden_size"]
        or config.get("num_hidden_layers") != model["layers"]
    ):
        raise ValueError("Lensの形式・次元・層数がモデルと一致しません")
    if config.get("observatory_model_fingerprint") != model["fingerprint"]:
        raise ValueError(
            "Lensの学習対象を確認するためconfig.jsonのobservatory_model_fingerprintに対象モデルのfingerprintを設定してください"
        )
    path = root / "params.safetensors"
    if not path.is_file():
        path = root / "params.pt"
    limit = min(512 * 2**20, budget_mib * 2**20)
    if not path.is_file() or path.stat().st_size > limit or model["hidden_size"] ** 2 * 16 > limit:
        raise ValueError("Lens重みがないか、RAM予算・512MiBの上限を超えます")
    try:
        if path.suffix == ".safetensors":
            state = load_file(str(path), device="cpu")
        else:
            if not zipfile.is_zipfile(path):
                raise ValueError("古い非ZIP形式のparams.ptは非対応です。safetensorsへ変換してください")
            with zipfile.ZipFile(path) as z:
                if sum(i.file_size for i in z.infolist()) > limit:
                    raise ValueError("展開後のLens容量が上限を超えます")
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        raise ValueError(f"安全なLens重みとして読み込めません: {type(exc).__name__}") from exc
    expected = {}
    for i in range(model["layers"]):
        expected[f"{i}.weight"] = (model["hidden_size"], model["hidden_size"])
        if config.get("bias", True):
            expected[f"{i}.bias"] = (model["hidden_size"],)
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Lensの層別パラメーターが不足または未対応です")
    for key, shape in expected.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype not in {torch.float32, torch.float16, torch.bfloat16}
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Lensの形状・dtype・有限値が不正です: {key}")
    fingerprint = hashlib.sha256((digest(config_path) + digest(path)).encode()).hexdigest()
    return config, state, fingerprint


def register_lens(store, model, path, name=""):
    config, _, fingerprint = read_checkpoint(path, model, store.settings["ram_budget_mib"])
    root = str(Path(path).expanduser().resolve())
    registration = dict(
        id=hashlib.sha256(root.encode()).hexdigest()[:16],
        path=root,
        name=name or Path(root).name,
        model=model,
        model_fingerprint=model["fingerprint"],
        fingerprint=fingerprint,
        format="AlignmentResearch linear_tuned_lens",
        training_info=str(config.get("training_info", "学習情報の記載なし。提供された重みを使用。")),
        source="local checkpoint supplied by user; training quality not independently certified",
    )
    store.put("lenses", registration["id"], registration)
    return registration


def apply_lens(store, registration, model_registration, stage, x):
    if registration["model_fingerprint"] != model_registration["fingerprint"]:
        raise ValueError("Lensとトレースのモデルfingerprintが異なります")
    config, state, fingerprint = read_checkpoint(
        registration["path"], model_registration, store.settings["ram_budget_mib"]
    )
    if fingerprint != registration["fingerprint"]:
        raise ValueError("Lensが登録時から変更されています。再登録してください")
    if stage == "embedding":
        index = 0
    else:
        parts = stage.split(".")
        if len(parts) != 3 or parts[0] != "layer" or parts[2] != "mlp" or not parts[1].isdigit():
            raise ValueError("Tuned LensはEmbeddingとBlock出力に対応し、Attention後には適用しません")
        index = int(parts[1]) + 1
    if index >= config["num_hidden_layers"]:
        return x, dict(index=index, translation="final block: translator not used")
    weight = state[f"{index}.weight"].to(device=x.device, dtype=x.dtype)
    bias = state.get(f"{index}.bias")
    if bias is not None:
        bias = bias.to(device=x.device, dtype=x.dtype)
    return x + torch.nn.functional.linear(x, weight, bias), dict(
        index=index, translation="h + affine(h)", checkpoint_fingerprint=fingerprint
    )
