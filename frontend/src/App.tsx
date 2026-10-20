import { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  Check,
  CircleDot,
  Database,
  FlaskConical,
  Loader2,
  Orbit,
  Settings as SettingsIcon,
  SlidersHorizontal,
  X,
} from 'lucide-react';
import { api, connect, type Job, type Model } from './api';
import Explorer from './Explorer';
import { Library, ModelManager, NewAnalysis, Settings } from './Pages';

const stages = ['実行確認', 'モデル準備', '入力処理', '内部解析', '座標変換', '表示準備'];
const terminal = ['completed', 'cancelled', 'failed'];

export default function App() {
  const [nav, setNav] = useState('analysis');
  const [ready, setReady] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [error, setError] = useState('');
  const [job, setJob] = useState<Job | null>(null);
  const [traceId, setTraceId] = useState('');
  const reload = useCallback(async () => {
    setModels(await api<Model[]>('models'));
  }, []);
  useEffect(() => {
    let active = true;
    connect()
      .then(async () => {
        await reload();
        const jobs = await api<Job[]>('analysis/jobs');
        if (!active) return;
        setReady(true);
        const running = jobs.find((j) => !terminal.includes(j.state));
        if (running) {
          setJob(running);
          setNav('progress');
        } else {
          const saved = localStorage.getItem('observatory-last-trace');
          if (saved) {
            try {
              await api(`traces/${saved}`);
              if (active) {
                setTraceId(saved);
                setNav('explorer');
              }
            } catch {
              localStorage.removeItem('observatory-last-trace');
            }
          }
        }
      })
      .catch((e) => setError(String(e)));
    return () => {
      active = false;
    };
  }, [reload]);
  function open(id: string) {
    setTraceId(id);
    setNav('explorer');
    localStorage.setItem('observatory-last-trace', id);
  }
  useEffect(() => {
    if (!job || terminal.includes(job.state)) return;
    const event = new EventSource(`/api/analysis/jobs/${job.id}/events`);
    event.onmessage = (message) => {
      const next = JSON.parse(message.data) as Job;
      setJob(next);
      if (terminal.includes(next.state)) {
        event.close();
        reload().catch((e) => setError(String(e)));
      }
    };
    event.onerror = () => {
      /* EventSource reconnects automatically; server owns the job. */
    };
    return () => event.close();
  }, [job?.id, reload]);
  const navigation = [
    { id: 'analysis', label: '新しい解析', icon: FlaskConical },
    { id: 'explorer', label: '3D Explorer', icon: Orbit },
    { id: 'library', label: 'Trace Library', icon: Database },
    { id: 'models', label: 'Model Manager', icon: SlidersHorizontal },
    { id: 'settings', label: '設定', icon: SettingsIcon },
  ];
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          href="#"
          className="brand"
          onClick={(e) => {
            e.preventDefault();
            setNav('analysis');
          }}
          aria-label="Transformer Observatory ホーム"
        >
          <Orbit size={32} />
          <span>
            Transformer
            <br />
            <b>Observatory</b>
          </span>
        </a>
        <div className="sidebar-section">WORKSPACE</div>
        <nav aria-label="主要ナビゲーション">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={nav === item.id ? 'nav-item active' : 'nav-item'}
              disabled={item.id === 'explorer' && !traceId}
              onClick={() => setNav(item.id)}
            >
              <item.icon size={18} />
              <span>{item.label}</span>
              {nav === item.id && <i />}
            </button>
          ))}
        </nav>
        {job && (
          <button className="running-job" onClick={() => setNav('progress')}>
            <Activity size={16} />
            <span>
              {terminal.includes(job.state)
                ? job.state === 'completed'
                  ? '解析が完了しました'
                  : 'ジョブを確認'
                : '解析を実行中'}
              <small>{job.detail}</small>
            </span>
          </button>
        )}
        <div className="sidebar-bottom">
          <span>
            <i className={ready ? 'live-dot' : 'offline-dot'} />
            {ready ? 'LOCAL BACKEND CONNECTED' : 'CONNECTING'}
          </span>
          <small>すべての推論は、このPCで。</small>
          <span className="version">v0.1.0 / RESEARCH PREVIEW</span>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <span>
            OBSERVATORY{' '}
            <span className="muted">/ {nav === 'progress' ? 'ANALYZING' : nav.toUpperCase()}</span>
          </span>
          <div>
            <CircleDot size={13} />
            <span>実測データで観測</span>
            <i />
            LOCAL SESSION
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button className="icon-button" title="通知を閉じる" onClick={() => setError('')}>
              <X size={16} />
            </button>
          </div>
        )}
        {!ready ? (
          <div className="empty-state">
            <Loader2 className="spin" />
            <p>ローカルバックエンドへ接続しています。</p>
            {error && (
              <button className="secondary" onClick={() => location.reload()}>
                再接続
              </button>
            )}
          </div>
        ) : (
          <>
            {nav === 'analysis' && (
              <NewAnalysis
                models={models}
                onJob={(j) => {
                  setJob(j);
                  setNav('progress');
                }}
                onError={setError}
              />
            )}
            {nav === 'models' && <ModelManager models={models} reload={reload} onError={setError} />}
            {nav === 'library' && <Library onOpen={open} onError={setError} />}
            {nav === 'settings' && <Settings onError={setError} />}
            {nav === 'explorer' && traceId && <Explorer id={traceId} onError={setError} />}
            {nav === 'progress' && job && (
              <div className="progress-page">
                <div className="eyebrow">OBSERVING THE TRANSFORMER</div>
                <div
                  className={`progress-ring ${terminal.includes(job.state) ? 'settled' : ''}`}
                  style={{ '--progress': `${job.percent * 3.6}deg` } as React.CSSProperties}
                >
                  <div>
                    <span className="progress-percent">
                      {job.state === 'completed' ? <Check size={48} /> : Math.floor(job.percent)}
                      {job.state !== 'completed' && <small>%</small>}
                    </span>
                    <span className="eyebrow">{job.state === 'completed' ? 'COMPLETE' : 'ESTIMATED'}</span>
                  </div>
                </div>
                <h1>
                  {job.state === 'completed'
                    ? '観測の準備ができました。'
                    : job.state === 'failed'
                      ? '解析を完了できませんでした。'
                      : job.state === 'cancelled'
                        ? '解析を中止しました。'
                        : 'ことばの変化を、記録しています。'}
                </h1>
                <p className="progress-detail" aria-live="polite">
                  {job.error || job.detail}
                </p>
                <div className="progress-stages">
                  {stages.map((s, i) => (
                    <div
                      key={s}
                      className={
                        i < job.stage || job.state === 'completed'
                          ? 'done'
                          : i === job.stage && !terminal.includes(job.state)
                            ? 'current'
                            : ''
                      }
                    >
                      <i>{i < job.stage || job.state === 'completed' ? <Check size={11} /> : null}</i>
                      <span>{s}</span>
                    </div>
                  ))}
                </div>
                <div className="button-row">
                  {!terminal.includes(job.state) ? (
                    <button
                      className="secondary"
                      disabled={job.state === 'cancelling'}
                      onClick={async () => {
                        try {
                          setJob(await api(`analysis/jobs/${job.id}/cancel`, 'POST'));
                        } catch (e) {
                          setError(String(e));
                        }
                      }}
                    >
                      解析を中止
                    </button>
                  ) : job.state === 'completed' ? (
                    <button className="primary" onClick={() => open(job.trace_id)}>
                      3D Explorerを開く
                      <ArrowRight size={17} />
                    </button>
                  ) : (
                    <button className="secondary" onClick={() => setNav('analysis')}>
                      <ArrowLeft size={16} />
                      入力に戻る
                    </button>
                  )}
                </div>
                {job.diagnostic && (
                  <details className="diagnostics">
                    <summary>詳細</summary>
                    <pre>{job.diagnostic}</pre>
                  </details>
                )}
                <p className="progress-note">
                  ブラウザを閉じても、バックエンドが起動している間は解析を継続します。
                </p>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
