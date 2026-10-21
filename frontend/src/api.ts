let csrf = '';
export async function connect() {
  const response = await fetch('/api/session', { credentials: 'same-origin' });
  if (!response.ok) throw new Error('ローカルバックエンドへ接続できません');
  const data = await response.json();
  csrf = data.token;
  return data;
}

export async function api<T = any>(
  path: string,
  method = 'GET',
  body?: unknown,
  headers?: HeadersInit,
): Promise<T> {
  const response = await fetch('/api/' + path, {
    method,
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      'X-Observatory-Token': csrf,
      ...headers,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let reason;
    try {
      reason = (await response.json()).detail;
    } catch {
      reason = response.statusText;
    }
    throw new Error(typeof reason === 'string' ? reason : JSON.stringify(reason));
  }
  return response.json();
}

export async function upload(file: File) {
  const r = await fetch('/api/storage/import', {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'X-Observatory-Token': csrf, 'Content-Type': 'application/zip' },
    body: file,
  });
  if (!r.ok) throw new Error((await r.json()).detail);
  return r.json();
}

export async function coordinates(id: string, view: string) {
  const r = await fetch(`/api/traces/${id}/coordinates?view=${view}`);
  if (!r.ok) throw new Error((await r.json()).detail);
  return new Float32Array(await r.arrayBuffer());
}

export const bytes = (value: number) =>
  value >= 2 ** 30 ? `${(value / 2 ** 30).toFixed(2)} GiB` : `${(value / 2 ** 20).toFixed(1)} MiB`;
export const number = (x: number | undefined | null) =>
  x == null ? '—' : Math.abs(x) < 0.001 && x !== 0 ? x.toExponential(2) : x.toFixed(3);

export type Token = {
  position: number;
  id: number;
  text: string;
  raw: string;
  generated: boolean;
  processed: boolean;
  step: number;
};
export type Model = {
  id: string;
  name: string;
  architecture: string;
  hidden_size: number;
  layers: number;
  heads: number;
  kv_heads: number;
  max_tokens: number;
  trained: boolean;
  capabilities: Record<string, any>;
  fingerprint: string;
  path?: string;
  available?: boolean;
  availability_reason?: string;
};
export type Manifest = {
  id: string;
  title: string;
  level: number;
  model: Model;
  tokens: Token[];
  stages: string[];
  status: string;
  projection: { basis: number[][]; explained_variance_ratio: number[]; algorithm: string };
  metric_names: string[];
  request: Record<string, any>;
  integrity: any;
  tensors: any[];
  checks: any[];
  generated_text?: string;
  retained: boolean;
  device: string;
  dtype: string;
};
export type Job = {
  id: string;
  trace_id: string;
  state: string;
  percent: number;
  stage: number;
  detail: string;
  error?: string;
  diagnostic?: string;
};
