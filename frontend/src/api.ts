let csrf = '';
export async function connect() {
  const response = await fetch('/api/session', { credentials: 'same-origin' });
  if (!response.ok) throw new Error('ローカルバックエンドへ接続できません');
  const data = await response.json();
  csrf = data.token;
  return data;
}

async function request<T = any>(
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

export type RuntimeStatus = {
  state: 'idle' | 'starting' | 'ready' | 'failed';
  error?: string | null;
  pid?: number;
};
type RuntimeView = { status: RuntimeStatus; operations: { id: number; path: string }[] };
let runtimeView: RuntimeView = { status: { state: 'idle' }, operations: [] };
const runtimeListeners = new Set<() => void>();
export const runtimeSnapshot = () => runtimeView;
export const subscribeRuntime = (listener: () => void) => {
  runtimeListeners.add(listener);
  return () => {
    runtimeListeners.delete(listener);
  };
};
function publishRuntime(change: Partial<RuntimeView>) {
  runtimeView = { ...runtimeView, ...change };
  runtimeListeners.forEach((listener) => listener());
}
let preparing: Promise<void> | undefined;
export function prepareRuntime(retry = false): Promise<void> {
  if (preparing) return preparing;
  preparing = (async () => {
    try {
      let status = await request<RuntimeStatus>('runtime/start', 'POST', { retry });
      publishRuntime({ status });
      while (status.state === 'starting') {
        await new Promise((resolve) => window.setTimeout(resolve, 700));
        status = await request<RuntimeStatus>('runtime');
        publishRuntime({ status });
      }
      if (status.state !== 'ready') throw new Error(status.error || '解析エンジンを準備できませんでした');
    } catch (error) {
      publishRuntime({ status: { state: 'failed', error: String(error) } });
      throw error;
    } finally {
      preparing = undefined;
    }
  })();
  return preparing;
}
let operationId = 0;
export async function api<T = any>(
  path: string,
  method = 'GET',
  body?: unknown,
  headers?: HeadersInit,
): Promise<T> {
  const heavy =
    method === 'POST' &&
    /^(analysis\/preflight|models\/register|models\/[^/]+\/relocate|traces\/compare|traces\/[^/]+\/lens|research\/studies|research\/lenses\/register)$/.test(
      path,
    );
  if (!heavy) return request<T>(path, method, body, headers);
  const id = ++operationId;
  publishRuntime({ operations: [...runtimeView.operations, { id, path }] });
  try {
    await prepareRuntime();
    return await request<T>(path, method, body, headers);
  } catch (error) {
    if (runtimeView.status.state === 'ready') {
      try {
        publishRuntime({ status: await request<RuntimeStatus>('runtime') });
      } catch {
        /* Keep the operation's original error. */
      }
    }
    throw error;
  } finally {
    publishRuntime({ operations: runtimeView.operations.filter((item) => item.id !== id) });
  }
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
  created_at?: number;
  elapsed_seconds?: number;
};
