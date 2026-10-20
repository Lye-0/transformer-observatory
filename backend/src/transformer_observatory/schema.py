from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnalysisRequest(BaseModel):
    model_id: str = "tiny-transformer"
    prompt: str = Field(min_length=1, max_length=100_000)
    title: str = Field(default="", max_length=160)
    level: int = Field(default=2, ge=1, le=4)
    generate: bool = False
    max_new_tokens: int = Field(default=16, ge=1, le=256)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    layers: list[int] = Field(default_factory=list, max_length=256)
    heads: list[int] = Field(default_factory=list, max_length=256)
    tensors: list[str] = Field(default_factory=list, max_length=64)
    attention: bool = False
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    projection: Literal["auto", "exact", "incremental"] = "auto"

    @model_validator(mode="after")
    def scope(self):
        if any(x < 0 for x in self.layers + self.heads):
            raise ValueError("Layer/Headは0以上で指定してください")
        allowed = {
            "q_linear",
            "k_linear",
            "v_linear",
            "q_norm",
            "k_norm",
            "q_pre_rope",
            "q",
            "k",
            "v",
            "head_output",
            "head_contribution",
            "attention_norm",
            "attention_update",
            "attention_weights",
            "attention_mask",
            "gate",
            "up",
            "activation",
            "product",
            "mlp_norm",
            "mlp_update",
        }
        if any(name not in allowed for name in self.tensors):
            raise ValueError("未対応のテンソル名が含まれています")
        self.layers = sorted(set(self.layers))
        self.heads = sorted(set(self.heads))
        self.tensors = sorted(set(self.tensors))
        return self


class ModelRegistration(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    name: str = Field(default="", max_length=160)


class TraceMetadataUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def tag_lengths(self):
        self.tags = sorted(set(t.strip() for t in self.tags if t.strip()))
        if any(len(t) > 40 for t in self.tags):
            raise ValueError("タグは40文字以内で指定してください")
        return self


class ComparisonRequest(BaseModel):
    left: str
    right: str


class SettingsUpdate(BaseModel):
    retention_days: int = Field(ge=1, le=3650)
    cache_limit_gib: float = Field(ge=0.1, le=4096)
    trace_limit_gib: float = Field(ge=0.05, le=1024)
    ram_budget_mib: int = Field(ge=64, le=65536)
    trace_root: str | None = Field(default=None, max_length=4096)


STAGES = ["実行確認", "モデル準備", "入力処理", "内部解析", "座標変換", "表示準備"]
TERMINAL = {"completed", "failed", "cancelled"}
