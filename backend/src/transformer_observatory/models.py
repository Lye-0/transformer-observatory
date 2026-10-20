import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

TINY_ID = 'tiny-transformer'


class ByteTokenizer:
    bos_token_id, eos_token_id = 256, 257

    def encode(self, text):
        return [self.bos_token_id, *text.encode('utf-8')]

    def decode(self, ids, **_):
        return bytes(i for i in ids if i < 256).decode('utf-8', errors='replace')

    def convert_ids_to_tokens(self, i):
        if i == 256:
            return '<BOS>'
        if i == 257:
            return '<EOS>'
        return chr(i) if 32 <= i < 127 else f'<{i:02X}>'


class TinyAttention(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.head_dim, self.num_heads, self.num_key_value_heads = d // heads, heads, heads
        self.scaling, self.sliding_window = self.head_dim**-0.5, None
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(d, d, bias=False) for _ in range(3)]
        self.o_proj = nn.Linear(d, d, bias=False)
        self.q_norm, self.k_norm = nn.Identity(), nn.Identity()

    def forward(self, hidden_states, **_):
        b, t, _ = hidden_states.shape
        q, k, v = [p(hidden_states).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj)]
        logits = q @ k.transpose(-1, -2) * self.scaling
        mask = torch.ones(t, t, device=q.device, dtype=torch.bool).triu(1)
        weights = logits.masked_fill(mask, float('-inf')).softmax(-1)
        z = (weights @ v).transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(z), weights


class TinyMLP(nn.Module):
    def __init__(self, d=64, intermediate=128):
        super().__init__()
        self.gate_proj, self.up_proj = [nn.Linear(d, intermediate, bias=False) for _ in range(2)]
        self.act_fn = nn.SiLU()
        self.down_proj = nn.Linear(intermediate, d, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(64)
        self.self_attn = TinyAttention()
        self.post_attention_layernorm = nn.LayerNorm(64)
        self.mlp = TinyMLP()

    def forward(self, x):
        r = x + self.self_attn(self.input_layernorm(x))[0]
        return r + self.mlp(self.post_attention_layernorm(r))


class TinyModel(nn.Module):
    """A real deterministic Transformer for numerical validation, not a trained LLM."""

    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(258, 64)
            self.model.layers = nn.ModuleList([TinyBlock() for _ in range(4)])
            self.model.norm = nn.LayerNorm(64)
            self.lm_head = nn.Linear(64, 258, bias=False)
        self.config = SimpleNamespace(model_type='observatory_tiny', hidden_size=64,
                                      num_hidden_layers=4, num_attention_heads=4,
                                      num_key_value_heads=4, max_position_embeddings=2048)

    def forward(self, input_ids, **_):
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)), past_key_values=None)


def capabilities(architecture, verified=False):
    state = '検証済み' if verified else '取得可能・未検証'
    return dict(architecture=architecture, levels=[1, 2, 3, 4], residual=state,
                attention=state, mlp=state, generation=state, quantization='非対応',
                interventions='非対応', tuned_lens='非対応')


def tiny_registration():
    return dict(id=TINY_ID, name='極小Transformer · 数値検証用', path=None,
                architecture='observatory_tiny', hidden_size=64, layers=4, heads=4,
                kv_heads=4, max_tokens=2048, trained=False,
                fingerprint='tiny-v1-seed1729-byte258-d64-l4',
                capabilities=capabilities('observatory_tiny', True))


def register_model(path, name=''):
    root = Path(path).expanduser().resolve()
    if not root.is_dir() or not (root / 'config.json').is_file():
        raise ValueError('config.jsonを含むローカルモデルフォルダーを指定してください')
    config_bytes = (root / 'config.json').read_bytes()
    config = json.loads(config_bytes)
    architecture = config.get('model_type')
    if architecture not in {'qwen3', 'gpt2'}:
        raise ValueError(f'未対応アーキテクチャ: {architecture}。現在はQwen3/GPT-2を対象にします')
    if config.get('quantization_config'):
        raise ValueError('量子化モデルの内部取得は未検証のため、現在は登録できません')
    if config.get('sliding_window') and config.get('use_sliding_window'):
        raise ValueError('Sliding-window Attentionの取得は未検証です')
    weight_files = sorted(root.glob('*.safetensors'))
    if not weight_files:
        raise ValueError('安全なsafetensors形式のモデル重みが必要です')
    # Hash actual weights and tokenizer artifacts, not just a directory name.
    fingerprint = hashlib.sha256(config_bytes)
    for f in weight_files + sorted(root.glob('*token*.json')) + sorted(root.glob('vocab.*')) + sorted(root.glob('merges.*')):
        fingerprint.update(f.name.encode())
        with f.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 2**20), b''):
                fingerprint.update(block)
    token_path = root / 'tokenizer.json'
    if not token_path.is_file() and not (root / 'vocab.json').is_file():
        raise ValueError('Tokenizerのファイルがありません')
    return dict(id=hashlib.sha256(str(root).encode()).hexdigest()[:16],
                name=name or root.name, path=str(root), architecture=architecture,
                fingerprint=fingerprint.hexdigest(), trained=True,
                hidden_size=config.get('hidden_size', config.get('n_embd')),
                layers=config.get('num_hidden_layers', config.get('n_layer')),
                heads=config.get('num_attention_heads', config.get('n_head')),
                kv_heads=config.get('num_key_value_heads', config.get('n_head')),
                max_tokens=config.get('max_position_embeddings', config.get('n_positions', 2048)),
                weight_bytes=sum(f.stat().st_size for f in weight_files),
                capabilities=capabilities(architecture))


def load_tokenizer(registration):
    if registration['id'] == TINY_ID:
        return ByteTokenizer()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(registration['path'], local_files_only=True, trust_remote_code=False)


def tokenize(tokenizer, prompt):
    ids = tokenizer.encode(prompt)
    return [dict(position=i, id=int(token_id), text=tokenizer.decode([token_id]),
                 raw=str(tokenizer.convert_ids_to_tokens(token_id)), generated=False, step=0,
                 processed=True) for i, token_id in enumerate(ids)]


def load_model(registration, device, dtype):
    if registration['id'] == TINY_ID:
        return TinyModel().to(device=device, dtype=dtype).eval()
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        registration['path'], local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, attn_implementation='eager',
    ).to(device).eval()
