import difflib

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA


def fit_projection(arrays, method="auto", budget_mib=512, seed=42):
    """One fixed basis over all checkpoints. Input iterable must be re-iterable."""
    total_rows, total_bytes, width = 0, 0, None
    for a in arrays():
        total_rows += len(a)
        total_bytes += a.nbytes
        width = a.shape[-1]
    if not total_rows:
        raise ValueError("射影するResidualがありません")
    if total_rows == 1:
        row = next(iter(arrays()))[0]
        return dict(
            basis=np.zeros((3, width)).tolist(),
            mean=row.tolist(),
            algorithm="degenerate-single-row",
            explained_variance_ratio=[0.0, 0.0, 0.0],
            seed=seed,
            sign_convention="largest absolute loading positive",
            fit_rows=1,
            source_space="residual",
            centered=True,
        )
    components = min(3, total_rows, width)
    algorithm = method
    if method == "auto":
        algorithm = "randomized" if total_bytes < budget_mib * 2**20 // 3 else "incremental"
    if components < 3:
        algorithm = "exact"
    if algorithm == "incremental":
        estimator = IncrementalPCA(n_components=3)
        batch_rows = max(6, min(512, budget_mib * 2**20 // max(width * 8 * 8, 1)))
        pending = np.empty((0, width), dtype=np.float32)
        for a in arrays():
            for begin in range(0, len(a), batch_rows):
                pending = np.concatenate((pending, a[begin : begin + batch_rows]))
                if len(pending) >= batch_rows + 3:
                    estimator.partial_fit(pending[:batch_rows])
                    pending = pending[batch_rows:]
        if len(pending):
            estimator.partial_fit(pending)
    else:
        if total_bytes > budget_mib * 2**20 // 2 and method == "exact":
            raise ValueError("Exact PCAのRAM見積もりが上限を超えます。auto/incrementalを使用してください")
        values = np.concatenate([np.asarray(a, dtype=np.float32) for a in arrays()])
        estimator = PCA(
            n_components=components,
            svd_solver="full" if algorithm == "exact" else "randomized",
            random_state=seed,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            estimator.fit(values)
    basis = np.zeros((3, width), dtype=np.float64)
    basis[:components] = estimator.components_
    for row in basis:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    ratios = np.nan_to_num(estimator.explained_variance_ratio_).tolist()
    ratios += [0.0] * (3 - len(ratios))
    return dict(
        basis=basis.tolist(),
        mean=estimator.mean_.tolist(),
        algorithm=algorithm,
        explained_variance_ratio=ratios,
        seed=seed,
        sign_convention="largest absolute loading positive",
        fit_rows=total_rows,
        source_space="residual",
        centered=True,
    )


def records(manifest):
    return [r for r in manifest["tensors"] if r["kind"] == "residual"]


def project_trace(store, writer, cancel):
    m = writer.manifest
    residuals = records(m)

    def arrays():
        for r in residuals:
            if cancel.is_set():
                from .capture import Cancelled

                raise Cancelled()
            yield store.load(m["id"], r)

    pca = fit_projection(
        arrays, m["request"]["projection"], store.settings["ram_budget_mib"], m["request"]["seed"]
    )
    m["projection"] = pca
    coords, disp, metrics = transform(store, m, pca)
    for a, kind in [(coords, "coordinates"), (disp, "displacement"), (metrics, "metrics")]:
        writer.write(a, kind=kind, stage="all", positions=list(range(len(coords))))
    validity = np.ones(metrics.shape, dtype=np.uint8)
    previous_norms = np.concatenate((metrics[:, :1, 0], metrics[:, :-1, 0]), axis=1)
    validity[:, :, 3] = previous_norms > 0
    validity[:, :, 4] = (metrics[:, :, 0] > 0) & (previous_norms > 0)
    validity[:, :, 6] = 0
    for stage, name in enumerate(m["stages"]):
        if name.endswith(".mlp") and stage > 0:
            validity[:, stage, 6] = (metrics[:, stage, 2] > 0) & (metrics[:, stage - 1, 2] > 0)
    writer.write(validity, kind="metric_validity", stage="all", positions=list(range(len(coords))))
    m["metric_names"] = [
        "norm",
        "displacement",
        "update",
        "relative_update",
        "cosine",
        "projected_update",
        "attention_mlp_cosine",
    ]
    m["projection"]["measurement"] = "保存された高次元Residualの共通線形射影"


def transform(store, manifest, pca):
    stages = manifest["stages"]
    total_tokens = sum(t["processed"] for t in manifest["tokens"])
    basis, mean = np.array(pca["basis"]), np.array(pca["mean"])
    coords = np.zeros((total_tokens, len(stages), 3), dtype=np.float32)
    disp = np.zeros_like(coords)
    metrics = np.zeros((total_tokens, len(stages), 7), dtype=np.float32)
    # At most a step's current/initial/attention values in RAM; no full d-dimensional history.
    current_step, initial, previous, attention_update = None, None, None, None
    for r in records(manifest):
        stage = stages.index(r["stage"])
        positions = r["positions"]
        x = np.asarray(store.load(manifest["id"], r), dtype=np.float64)
        if r["step"] != current_step:
            current_step, initial, previous = r["step"], x.copy(), x.copy()
            attention_update = np.zeros_like(x)
        delta, update = x - initial, x - previous
        projected = (x - mean) @ basis.T
        d_projected = delta @ basis.T
        coords[positions, stage] = projected
        disp[positions, stage] = d_projected
        norm = np.linalg.norm(x, axis=-1)
        prev_norm = np.linalg.norm(previous, axis=-1)
        update_norm = np.linalg.norm(update, axis=-1)
        cosine = np.sum(x * previous, axis=-1) / np.maximum(norm * prev_norm, 1e-30)
        cosine = np.clip(cosine, -1, 1)
        angle = np.zeros(len(x))
        if r["stage"].endswith(".attention"):
            attention_update = update.copy()
        if r["stage"].endswith(".mlp"):
            denom = np.linalg.norm(attention_update, axis=-1) * update_norm
            angle = np.divide(
                np.sum(attention_update * update, axis=-1),
                denom,
                out=np.zeros_like(denom),
                where=denom > 1e-30,
            )
        metrics[positions, stage] = np.stack(
            (
                norm,
                np.linalg.norm(delta, axis=-1),
                update_norm,
                update_norm / np.maximum(prev_norm, 1e-30),
                cosine,
                np.linalg.norm(update @ basis.T, axis=-1),
                np.clip(angle, -1, 1),
            ),
            -1,
        )
        previous = x
    return coords, disp, metrics


def compare(store, left_id, right_id):
    left, right = store.manifest(left_id), store.manifest(right_id)
    if left["status"] != "completed" or right["status"] != "completed":
        raise ValueError("完了トレースのみ比較できます")
    if left["model"]["fingerprint"] != right["model"]["fingerprint"] or left["stages"] != right["stages"]:
        raise ValueError("同一のモデル重み・Tokenizer・観測点を持つ実験が必要です")
    if not records(left) or not records(right):
        raise ValueError("共通射影の再計算には両実験のLevel 2以上のResidualが必要です")

    def arrays():
        for m in (left, right):
            for r in records(m):
                yield store.load(m["id"], r)

    pca = fit_projection(arrays, "auto", store.settings["ram_budget_mib"])
    lc, ld, _ = transform(store, left, pca)
    rc, rd, _ = transform(store, right, pca)
    a, b = [[t["id"] for t in m["tokens"] if t["processed"]] for m in (left, right)]
    alignment = []
    for tag, i, j, k, end in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        count = max(j - i, end - k)
        for offset in range(count):
            x, y = (i + offset if i + offset < j else None), (k + offset if k + offset < end else None)
            alignment.append(
                dict(left=x, right=y, match=x is not None and y is not None and a[x] == b[y], operation=tag)
            )
    # Difference metrics are computed in the original high-dimensional residual space.
    differences = []
    left_lookup = {(r["stage"], p): (r, idx) for r in records(left) for idx, p in enumerate(r["positions"])}
    right_lookup = {(r["stage"], p): (r, idx) for r in records(right) for idx, p in enumerate(r["positions"])}
    for stage in left["stages"]:
        left_arrays = {r["file"]: store.load(left_id, r) for r in records(left) if r["stage"] == stage}
        right_arrays = {r["file"]: store.load(right_id, r) for r in records(right) if r["stage"] == stage}
        for pair in alignment:
            if pair["left"] is None or pair["right"] is None:
                continue
            lr, li = left_lookup[stage, pair["left"]]
            rr, ri = right_lookup[stage, pair["right"]]
            x = np.asarray(left_arrays[lr["file"]][li], dtype=np.float64)
            y = np.asarray(right_arrays[rr["file"]][ri], dtype=np.float64)
            differences.append(
                dict(
                    stage=stage,
                    **pair,
                    distance=float(np.linalg.norm(x - y)),
                    cosine=float(np.clip(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1))
                    if np.linalg.norm(x) * np.linalg.norm(y) > 0
                    else None,
                )
            )
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    prefix_error = max(
        (d["distance"] for d in differences if d["left"] < prefix and d["right"] == d["left"]), default=0
    )
    return dict(
        left=left_id,
        right=right_id,
        projection=pca,
        stages=left["stages"],
        alignment=alignment,
        left_tokens=left["tokens"],
        right_tokens=right["tokens"],
        coordinates={"left": lc.tolist(), "right": rc.tolist()},
        displacement={"left": ld.tolist(), "right": rd.tolist()},
        differences=differences,
        settings_differ=any(
            left.get(key) != right.get(key)
            for key in ["device", "dtype", "capture_backend", "observation_definition"]
        ),
        common_prefix_tokens=prefix,
        common_prefix_max_distance=prefix_error,
    )
