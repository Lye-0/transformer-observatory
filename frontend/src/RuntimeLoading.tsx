import { lazy, Suspense, useSyncExternalStore } from 'react';
import { prepareRuntime, runtimeSnapshot, subscribeRuntime } from './api';
import './runtime-loading.css';

const HelixThreadLoader = lazy(() => import('./components/helix-thread-loader/HelixThreadLoader'));
function LoadingVisual({ content, className }: { content: string; className?: string }) {
  return (
    <Suspense fallback={<p role="status">{content}</p>}>
      <HelixThreadLoader className={className} label="" content={content} />
    </Suspense>
  );
}

export function RuntimeLoading({
  scope,
}: {
  scope: 'models' | 'analysis' | 'research' | 'lens' | 'compare';
}) {
  const { status, operations } = useSyncExternalStore(subscribeRuntime, runtimeSnapshot);
  const active = operations.some(({ path }) => {
    if (scope === 'lens') return /^traces\/[^/]+\/lens$/.test(path);
    if (scope === 'compare') return path === 'traces/compare';
    return path.startsWith(scope + '/');
  });
  if (!active) return null;
  return (
    <div className="runtime-loading" aria-busy="true">
      <LoadingVisual
        content={status.state === 'ready' ? '処理しています…' : '解析エンジンを準備しています…'}
      />
      {status.state !== 'ready' && <p>初回は時間がかかる場合があります。他の画面は引き続き利用できます。</p>}
    </div>
  );
}

export function RuntimeFailure() {
  const { status } = useSyncExternalStore(subscribeRuntime, runtimeSnapshot);
  if (status.state !== 'failed') return null;
  return (
    <div className="runtime-failure" role="status">
      <span>解析エンジンを準備できませんでした。閲覧は引き続き利用できます。</span>
      <button
        className="secondary compact"
        onClick={() => {
          prepareRuntime(true).catch(() => {});
        }}
      >
        準備を再試行
      </button>
      <details>
        <summary>詳細</summary>
        {status.error}
      </details>
    </div>
  );
}

export function AnalysisPreparing() {
  return <LoadingVisual className="analysis-preparing" content="解析エンジンを準備しています…" />;
}
