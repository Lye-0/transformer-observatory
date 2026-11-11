"""Model catalog metadata, without importing inference libraries."""

TINY_ID = "tiny-transformer"


def capabilities(architecture, verified=False):
    state = "検証済み" if verified else "取得可能・未検証"
    return dict(
        architecture=architecture,
        levels=[1, 2, 3, 4],
        residual=state,
        attention=state,
        mlp=state,
        generation=state,
        quantization="非対応",
        interventions="対応（Prefill・境界限定）",
        tuned_lens="学習済み重みの登録が必要",
    )


def tiny_registration():
    return dict(
        id=TINY_ID,
        name="極小Transformer · 数値検証用",
        path=None,
        architecture="observatory_tiny",
        hidden_size=64,
        layers=4,
        heads=4,
        kv_heads=4,
        max_tokens=2048,
        trained=False,
        fingerprint="tiny-v1-seed1729-byte258-d64-l4",
        capabilities=capabilities("observatory_tiny", True),
    )
