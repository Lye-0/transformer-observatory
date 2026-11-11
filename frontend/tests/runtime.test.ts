import assert from 'node:assert/strict';
import test from 'node:test';
import { api, prepareRuntime, runtimeSnapshot, subscribeRuntime } from '../src/api.ts';

test('concurrent operations share preparation, clear their loaders, and recover after failure', async () => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  let starts = 0;
  let heavyCalls = 0;
  let failed = false;
  let crashDuringRequest = false;
  let state = 'starting';
  const notifications: number[] = [];
  const unsubscribe = subscribeRuntime(() => notifications.push(runtimeSnapshot().operations.length));
  globalThis.window = { setTimeout } as unknown as Window & typeof globalThis;
  globalThis.fetch = async (url, options) => {
    let data: unknown;
    if (url === '/api/runtime/start') {
      starts++;
      if (JSON.parse(String(options?.body)).retry) {
        failed = false;
        state = 'ready';
      }
      data = { state: failed ? 'failed' : state, error: failed ? 'native import failed' : null };
    } else if (url === '/api/runtime') {
      state = 'ready';
      data = { state: failed ? 'failed' : state };
    } else {
      assert.equal(state, 'ready');
      heavyCalls++;
      if (crashDuringRequest) {
        failed = true;
        return new Response(JSON.stringify({ detail: 'worker disconnected' }), { status: 503 });
      }
      data = { result: 'verified' };
    }
    return new Response(JSON.stringify(data), { status: 200 });
  };
  try {
    const one = api('analysis/preflight', 'POST', {});
    const two = api('models/register', 'POST', {});
    assert.equal(runtimeSnapshot().operations.length, 2);
    assert.equal(heavyCalls, 0);
    await Promise.all([one, two]);
    assert.equal(starts, 1);
    assert.equal(heavyCalls, 2);
    assert.equal(runtimeSnapshot().operations.length, 0);
    failed = true;
    await assert.rejects(api('traces/compare', 'POST', {}), /native import failed/);
    assert.equal(runtimeSnapshot().status.state, 'failed');
    assert.equal(runtimeSnapshot().operations.length, 0);
    await prepareRuntime(true);
    await api('traces/example/lens', 'POST', {});
    assert.equal(runtimeSnapshot().status.state, 'ready');
    assert.equal(heavyCalls, 3);
    assert.ok(notifications.includes(2));
    crashDuringRequest = true;
    await assert.rejects(api('traces/example/lens', 'POST', {}), /worker disconnected/);
    assert.equal(runtimeSnapshot().status.state, 'failed');
    assert.equal(runtimeSnapshot().operations.length, 0);
  } finally {
    unsubscribe();
    globalThis.fetch = originalFetch;
    globalThis.window = originalWindow;
  }
});
