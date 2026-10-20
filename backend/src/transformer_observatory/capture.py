"""Hooks observe actual operations. Residual checkpoints are never inferred from 3D."""

import numpy as np
import torch


class Cancelled(Exception):
    pass


def cpu(t):
    if isinstance(t, (tuple, list)):
        t = t[0]
    return t.detach().to(device='cpu', dtype=torch.float32).numpy().copy()


class Capture:
    def __init__(self, model, writer, request, cancellation, progress):
        self.model, self.writer, self.request = model, writer, request
        self.cancellation, self.progress = cancellation, progress
        self.handles, self.scratch, self.history = [], {}, {}
        self.step, self.positions, self.save_slice = 0, [], slice(None)
        self.checks = []
        self.arch = model.config.model_type
        self.layers = model.transformer.h if self.arch == 'gpt2' else model.model.layers
        self.selected = set(request.layers or range(len(self.layers)))
        if any(x >= len(self.layers) for x in self.selected):
            raise ValueError('取得対象Layerがモデルの範囲外です')
        head_count = model.config.n_head if self.arch == 'gpt2' else model.config.num_attention_heads
        if any(x >= head_count for x in request.heads):
            raise ValueError('取得対象Headがモデルの範囲外です')
        self.heads = request.heads or list(range(head_count))
        self.dtype = str(next(model.parameters()).dtype)
        self.atol = 2e-5 if 'float32' in self.dtype else 0.05
        self.rtol = 2e-4 if 'float32' in self.dtype else 0.04

    def check(self, label, actual, expected, layer):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f'{label}の形状が不一致: {a.shape} / {b.shape}')
        error = float(np.max(np.abs(a - b), initial=0))
        passed = bool(np.allclose(a, b, atol=self.atol, rtol=self.rtol))
        self.checks.append(dict(name=label, layer=layer, step=self.step, max_abs_error=error,
                                atol=self.atol, rtol=self.rtol, passed=passed))
        if not passed:
            raise ValueError(f'数値整合性チェックに失敗: {label} Layer {layer}, error={error:.6g}')

    def write(self, a, kind, stage, layer, space='residual', heads=None):
        self.writer.write(a, kind=kind, stage=stage, layer=layer, step=self.step,
                          positions=self.positions, space=space, heads=heads)

    def residual(self, a, stage, layer):
        self.write(a[0, self.save_slice], 'residual', stage, layer)

    def detail(self, a, name, layer, space, heads=None):
        if self.request.level >= 3 and layer in self.selected:
            self.write(a, 'tensor', name, layer, space, heads)

    def set_step(self, step, positions, recomputed=False):
        self.step, self.positions = step, list(positions)
        self.save_slice = slice(-1, None) if recomputed else slice(None)

    def install(self):
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self.block_before(index)))
            self.handles.append(layer.register_forward_hook(self.block_after(index)))
            norm = layer.ln_2 if self.arch == 'gpt2' else layer.post_attention_layernorm
            attn = layer.attn if self.arch == 'gpt2' else layer.self_attn
            self.handles.append(norm.register_forward_pre_hook(self.after_attention(index)))
            self.handles.append(attn.register_forward_hook(self.attention_after(index)))
            self.handles.append(layer.mlp.register_forward_hook(self.mlp_after(index)))
            if self.request.level >= 3 and index in self.selected:
                self.install_details(index, layer, attn)
        return self

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.scratch.clear()
        self.history.clear()

    def block_before(self, index):
        def hook(module, args):
            if self.cancellation.is_set():
                raise Cancelled('解析を中止しました')
            self.scratch[index] = {'x': cpu(args[0])}
            if index == 0:
                self.residual(self.scratch[index]['x'], 'embedding', -1)
            self.progress(index, len(self.layers), self.step)
        return hook

    def attention_after(self, index):
        def hook(module, args, out):
            self.scratch[index]['attn'] = cpu(out)
            if self.request.level >= 3 and index in self.selected:
                if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                    self.scratch[index]['weights'] = cpu(out[1])
                    if self.request.attention:
                        self.detail(cpu(out[1])[0, self.heads, self.save_slice], 'attention_weights',
                                    index, 'attention_coefficients', self.heads)
                self.detail(cpu(out)[0, self.save_slice], 'attention_update', index, 'residual')
        return hook

    def after_attention(self, index):
        def hook(module, args):
            r = cpu(args[0])
            s = self.scratch[index]
            self.check('attention_residual_add', r, s['x'] + s['attn'], index)
            s['r'] = r
            self.residual(r, f'layer.{index}.attention', index)
        return hook

    def mlp_after(self, index):
        def hook(module, args, out):
            s = self.scratch[index]
            s['mlp'] = cpu(out)
            self.detail(cpu(out)[0, self.save_slice], 'mlp_update', index, 'residual')
            self.detail(cpu(args[0])[0, self.save_slice], 'mlp_norm', index, 'residual_normalized')
        return hook

    def block_after(self, index):
        def hook(module, args, out):
            x = cpu(out)
            s = self.scratch[index]
            self.check('mlp_residual_add', x, s['r'] + s['mlp'], index)
            self.residual(x, f'layer.{index}.mlp', index)
            if 'head_output' in s:
                self.validate_attention(index, s, module)
            if 'product' in s and 'gate' in s and 'up' in s:
                expected = s['gate'] / (1 + np.exp(-np.clip(s['gate'], -80, 80))) * s['up']
                self.check('swiglu_product', s['product'], expected, index)
            del self.scratch[index]
        return hook

    def install_details(self, index, layer, attn):
        def save(name, space, pre=False):
            def hook(module, args, out=None):
                a = cpu(args[0] if pre else out)
                self.scratch[index][name] = a
                self.detail(a[0, self.save_slice], name, index, space)
            return hook

        self.handles.append(attn.register_forward_pre_hook(self.attention_before(index), with_kwargs=True))
        if self.arch == 'gpt2':
            self.handles.append(attn.c_attn.register_forward_hook(save('qkv_linear', 'qkv_linear')))
            self.handles.append(attn.c_proj.register_forward_pre_hook(save('head_output', 'head_values', True)))
            self.handles.append(layer.mlp.c_fc.register_forward_hook(save('up', 'mlp_intermediate')))
            self.handles.append(layer.mlp.c_proj.register_forward_pre_hook(save('activation', 'mlp_intermediate', True)))
        else:
            for name in ['q', 'k', 'v']:
                self.handles.append(getattr(attn, name + '_proj').register_forward_hook(
                    save(name + '_linear', name + '_linear')))
            self.handles.append(attn.o_proj.register_forward_pre_hook(save('head_output', 'head_values', True)))
            for name in ['gate', 'up']:
                self.handles.append(getattr(layer.mlp, name + '_proj').register_forward_hook(
                    save(name, 'mlp_intermediate')))
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(save('product', 'mlp_intermediate', True)))
            self.handles.append(layer.mlp.act_fn.register_forward_hook(save('activation', 'mlp_intermediate')))
            if self.arch == 'qwen3':
                self.handles.append(attn.q_norm.register_forward_hook(save('q_norm', 'query_pre_rope')))
                self.handles.append(attn.k_norm.register_forward_hook(save('k_norm', 'key_pre_rope')))

    def attention_before(self, index):
        def hook(module, args, kwargs):
            x = kwargs.get('hidden_states', args[0] if args else None)
            self.detail(cpu(x)[0, self.save_slice], 'attention_norm', index, 'residual_normalized')
            embeddings = kwargs.get('position_embeddings')
            if embeddings:
                self.scratch[index]['rope'] = tuple(cpu(t) for t in embeddings)
        return hook

    def validate_attention(self, index, s, block):
        attn = block.attn if self.arch == 'gpt2' else block.self_attn
        heads = self.model.config.n_head if self.arch == 'gpt2' else self.model.config.num_attention_heads
        d_h = s['head_output'].shape[-1] // heads
        if self.arch == 'gpt2':
            q, k, v = np.split(s['qkv_linear'], 3, axis=-1)
            q, k, v = [a.reshape(1, -1, heads, d_h).transpose(0, 2, 1, 3) for a in (q, k, v)]
            w_o = cpu(attn.c_proj.weight).T
        else:
            q = s.get('q_norm', s['q_linear'].reshape(1, -1, heads, d_h)).transpose(0, 2, 1, 3)
            kv_heads = s['k_linear'].shape[-1] // d_h
            k = s.get('k_norm', s['k_linear'].reshape(1, -1, kv_heads, d_h)).transpose(0, 2, 1, 3)
            v = s['v_linear'].reshape(1, -1, kv_heads, d_h).transpose(0, 2, 1, 3)
            self.detail(q[0, self.heads, self.save_slice].transpose(1, 0, 2), 'q_pre_rope', index,
                        'query_pre_rope', self.heads)
            if 'rope' in s:
                cos, sin = [a[:, None] for a in s['rope']]
                def rotate(a):
                    u, v = np.split(a, 2, axis=-1)
                    return np.concatenate((-v, u), axis=-1)
                q, k = q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
            w_o = cpu(attn.o_proj.weight)
        self.detail(q[0, self.heads, self.save_slice].transpose(1, 0, 2), 'q', index, 'query', self.heads)
        self.detail(k[0, :, self.save_slice].transpose(1, 0, 2), 'k', index, 'key', list(range(k.shape[1])))
        self.detail(v[0, :, self.save_slice].transpose(1, 0, 2), 'v', index, 'value', list(range(v.shape[1])))
        if self.step > 0 and self.arch != 'observatory_tiny':
            old_k, old_v = self.history[index]
            k, v = np.concatenate((old_k, k), axis=2), np.concatenate((old_v, v), axis=2)
        self.history[index] = (k, v)
        groups = heads // k.shape[1]
        keys, values = np.repeat(k, groups, axis=1), np.repeat(v, groups, axis=1)
        logits = q @ keys.transpose(0, 1, 3, 2) / np.sqrt(d_h)
        start = k.shape[2] - q.shape[2]
        mask = np.arange(k.shape[2])[None, :] > np.arange(start, start + q.shape[2])[:, None]
        logits = np.where(mask[None, None], -np.inf, logits)
        probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        z = (probs @ values).transpose(0, 2, 1, 3).reshape(s['head_output'].shape)
        self.check('attention_qkv_reconstruction', s['head_output'], z, index)
        if 'weights' in s:
            self.check('attention_coefficients', s['weights'], probs, index)
        z_heads = s['head_output'][0, self.save_slice].reshape(-1, heads, d_h)
        # Each head is mapped through its actual W_O block before entering residual space.
        contribution = np.einsum('thk,dhk->thd', z_heads[:, self.heads],
                                 w_o.reshape(w_o.shape[0], heads, d_h)[:, self.heads])
        self.detail(contribution, 'head_contribution', index, 'residual', self.heads)
