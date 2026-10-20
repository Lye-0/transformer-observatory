"""Validate portable v1 traces before accepting their scientific dimensions."""

import re

import numpy as np


def validate_manifest(m):
    tokens, stages, tensors = m["tokens"], m["stages"], m["tensors"]
    count, checkpoints = len(tokens), len(stages)
    if (
        not 1 <= count <= 32768
        or checkpoints != 2 * m["model"]["layers"] + 1
        or len(set(stages)) != checkpoints
    ):
        raise ValueError("トークン数または観測点の定義が不正です")
    if not all(
        t["position"] == i and t["processed"] and isinstance(t["id"], int) and t["step"] >= 0
        for i, t in enumerate(tokens)
    ):
        raise ValueError("完了トレースのトークン位置・処理状態が不正です")
    expected = {
        "coordinates": [count, checkpoints, 3],
        "displacement": [count, checkpoints, 3],
        "metrics": [count, checkpoints, 7],
        "metric_validity": [count, checkpoints, 7],
    }
    files, summaries, residuals = set(), set(), set()
    for r in tensors:
        if r["file"] in files or not re.fullmatch(r"[0-9a-f]{64}", r["sha256"]):
            raise ValueError("重複ファイルまたは不正なハッシュ")
        files.add(r["file"])
        if r["dtype"] not in {"float32", "float64", "float16", "uint8", "int32", "int64"}:
            raise ValueError("未対応のテンソルdtype")
        if r["kind"] in expected:
            if r["shape"] != expected[r["kind"]] or r["kind"] in summaries:
                raise ValueError("描画・指標データの形状または件数が不正です")
            summaries.add(r["kind"])
        elif r["kind"] in {"residual", "tensor"}:
            positions = r["positions"]
            if len(set(positions)) != len(positions) or any(
                p < 0 or p >= count or tokens[p]["step"] != r["step"] for p in positions
            ):
                raise ValueError("取得テンソルの位置・生成Stepが不正です")
            axis = 1 if r["stage"] == "attention_weights" else 0
            if len(positions) != r["shape"][axis]:
                raise ValueError("取得位置数とテンソル形状が一致しません")
            if r["kind"] == "residual":
                if (
                    r["stage"] not in stages
                    or len(r["shape"]) != 2
                    or r["shape"][1] != m["model"]["hidden_size"]
                ):
                    raise ValueError("Residualの空間・形状が不正です")
                for p in positions:
                    key = (r["stage"], p)
                    if key in residuals:
                        raise ValueError("Residualが重複しています")
                    residuals.add(key)
        else:
            raise ValueError("未対応のテンソル種別")
    if not {"coordinates", "displacement", "metrics"}.issubset(summaries):
        raise ValueError("完了トレースの描画データが不足しています")
    if m["level"] >= 2 and len(residuals) != count * checkpoints:
        raise ValueError("完了トレースのResidualに欠落があります")


def validate_finite(array):
    flat = np.ravel(array, order="K")
    for begin in range(0, flat.size, 262144):
        if not np.isfinite(flat[begin : begin + 262144]).all():
            raise ValueError("非有限値のテンソルはインポートできません")
