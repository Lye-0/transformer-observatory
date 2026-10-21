import { useEffect, useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowUp,
  Check,
  ChevronRight,
  Cpu,
  FolderOpen,
  Loader2,
  Plus,
  RefreshCw,
  Search,
  X,
} from 'lucide-react';
import { api, bytes, type Model } from './api';
import Select from './Select';

type Candidate = {
  path: string;
  name: string;
  architecture: string | null;
  status: string;
  reason: string;
  weight_bytes: number;
  layers?: number;
  hidden_size?: number;
};
type SearchConfig = {
  backend_os: string;
  roots: { label: string; path: string; exists: boolean }[];
  custom_paths: string[];
};
type Scan = {
  id: string;
  state: string;
  candidates: Candidate[];
  scanned: number;
  current: string;
  warnings: string[];
  truncated: boolean;
  error?: string;
};
type Props = {
  models: Model[];
  reload: () => Promise<void>;
  onError: (message: string) => void;
  onReturn: () => void;
  onRegistered: (model: Model) => Promise<void>;
};

export default function ModelManager({ models, reload, onError, onReturn, onRegistered }: Props) {
  const [path, setPath] = useState('');
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [scanError, setScanError] = useState('');
  const [inspection, setInspection] = useState<Candidate>();
  const [config, setConfig] = useState<SearchConfig>();
  const [scan, setScan] = useState<Scan>();
  const [searchPath, setSearchPath] = useState('');
  const [saving, setSaving] = useState(false);
  const [browser, setBrowser] = useState<'model' | 'search'>();
  const [relocating, setRelocating] = useState<Model>();
  const input = useRef<HTMLInputElement>(null);
  const mounted = useRef(true);
  const scanning = scan?.state === 'running';

  useEffect(() => {
    let active = true;
    mounted.current = true;
    api<SearchConfig>('models/discover-local')
      .then(async (result) => {
        if (!active) return;
        setConfig(result);
        const started = await api<Scan>('models/discovery', 'POST');
        if (active) setScan(started);
      })
      .catch((e) => {
        if (active) setScanError(String(e));
      });
    return () => {
      active = false;
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!scan || scan.state !== 'running') return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const result = await api<Scan>(`models/discovery/${scan.id}`);
        if (!active) return;
        setScan(result);
        if (result.state === 'running') timer = setTimeout(poll, 500);
      } catch (e) {
        if (active) {
          setScanError(String(e));
          setScan((old) => (old ? { ...old, state: 'failed' } : old));
        }
      }
    };
    timer = setTimeout(poll, 300);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [scan?.id, scan?.state]);
  useEffect(() => {
    setInspection(undefined);
    setMessage('');
  }, [path]);

  async function startScan() {
    setScanError('');
    try {
      setScan(await api<Scan>('models/discovery', 'POST'));
    } catch (e) {
      setScanError(String(e));
    }
  }
  async function savePaths(paths: string[]) {
    setSaving(true);
    try {
      setConfig(await api<SearchConfig>('models/search-paths', 'PUT', { paths }));
      setSearchPath('');
      await startScan();
    } catch (e) {
      setScanError(String(e));
    } finally {
      setSaving(false);
    }
  }
  async function inspect() {
    setBusy(true);
    setMessage('');
    try {
      setInspection(await api<Candidate>('models/inspect', 'POST', { path }));
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function register() {
    setBusy(true);
    setMessage('');
    try {
      const model = await api<Model>(
        relocating ? `models/${relocating.id}/relocate` : 'models/register',
        'POST',
        { path, name },
      );
      if (mounted.current) await onRegistered(model);
      else await reload();
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }
  function choose(candidate: Candidate) {
    setPath(candidate.path);
    setName(candidate.name);
    input.current?.focus();
  }
  return (
    <div className="page model-manager">
      <button className="text-button model-back" onClick={onReturn}>
        <ArrowLeft size={15} />
        解析に戻る
      </button>
      <div className="eyebrow">MODEL MANAGER</div>
      <h1>観測するモデル。</h1>
      <p className="page-intro">
        この{config?.backend_os || 'バックエンド'}環境に保存したモデルを検出し、解析に使うモデルを登録します。
      </p>

      <section className="panel model-registration" aria-labelledby="model-register-title">
        <div className="panel-heading">
          <span id="model-register-title">
            <FolderOpen size={17} />
            {relocating ? `${relocating.name}の保存場所を再指定` : '保存場所から登録'}
          </span>
        </div>
        <p className="subtle-note" id="model-path-help">
          config.json・Tokenizer・safetensorsの重みを含むフォルダーを指定してください。
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!busy && path.trim()) void register();
          }}
        >
          <label className="field" htmlFor="model-folder-path">
            モデルフォルダーの絶対パス
          </label>
          <div className="model-path-row">
            <input
              ref={input}
              disabled={busy}
              id="model-folder-path"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              aria-describedby="model-path-help"
            />
            <button type="button" className="secondary" disabled={busy} onClick={() => setBrowser('model')}>
              <FolderOpen size={16} />
              フォルダーを選択
            </button>
          </div>
          <label className="field">
            表示名（任意）
            <input value={name} disabled={busy} onChange={(e) => setName(e.target.value)} />
          </label>
          {inspection && (
            <div
              className={`model-inspection ${inspection.status === 'available' ? 'good' : ''}`}
              role="status"
            >
              <strong>{inspection.name}</strong>
              <span>{inspection.reason}</span>
              {inspection.status === 'available' && (
                <small>
                  {inspection.architecture} · {inspection.layers} blocks · {bytes(inspection.weight_bytes)}
                </small>
              )}
            </div>
          )}
          {message && (
            <p className="model-error" role="alert">
              {message}
            </p>
          )}
          <div className="form-actions">
            <button type="button" className="secondary" disabled={!path.trim() || busy} onClick={inspect}>
              <Search size={16} />
              内容を確認
            </button>
            <button className="primary" disabled={!path.trim() || busy}>
              {busy ? <Loader2 className="spin" size={16} /> : <Plus size={16} />}
              {busy ? '確認・重みの照合中…' : relocating ? '保存場所を更新して使う' : '登録して解析に使う'}
            </button>
            {relocating && (
              <button
                type="button"
                className="text-button"
                disabled={busy}
                onClick={() => {
                  setRelocating(undefined);
                  setPath('');
                  setName('');
                }}
              >
                再指定をやめる
              </button>
            )}
          </div>
        </form>
        <p className="subtle-note">
          現在はQwen3／GPT-2の非量子化safetensorsに対応。登録時にTokenizerと重みを照合します。モデルの大きさにより時間がかかります。
        </p>
      </section>

      <section className="panel model-discovery" aria-labelledby="model-discovery-title">
        <div className="panel-heading">
          <span id="model-discovery-title">
            <Search size={17} />
            ローカルモデルを検出
          </span>
          {scanning ? (
            <button
              className="secondary"
              onClick={async () => {
                try {
                  setScan(await api<Scan>(`models/discovery/${scan.id}/cancel`, 'POST'));
                } catch (e) {
                  setScanError(String(e));
                }
              }}
            >
              <X size={15} />
              検出を中止
            </button>
          ) : (
            <button className="secondary" disabled={saving || !config} onClick={startScan}>
              <RefreshCw size={15} />
              再検出
            </button>
          )}
        </div>
        <details className="model-search-settings">
          <summary>
            探索する場所 <span>{config?.roots.length || 0}か所</span>
          </summary>
          <ul className="model-root-list">
            {config?.roots.map((root) => (
              <li key={root.path}>
                <div>
                  <strong>{root.label}</strong>
                  <code>{root.path}</code>
                  <small>{root.exists ? 'フォルダーあり' : '現在この場所は見つかりません'}</small>
                </div>
                {config.custom_paths.includes(root.path) && (
                  <button
                    className="icon-button"
                    disabled={saving || scanning}
                    aria-label={`探索先を削除: ${root.path}`}
                    onClick={() => savePaths(config.custom_paths.filter((p) => p !== root.path))}
                  >
                    <X size={15} />
                  </button>
                )}
              </li>
            ))}
          </ul>
          <label className="field" htmlFor="model-search-path">
            探索先の絶対パスを追加
          </label>
          <div className="model-path-row">
            <input
              id="model-search-path"
              value={searchPath}
              onChange={(e) => setSearchPath(e.target.value)}
            />
            <button className="secondary" disabled={saving || scanning} onClick={() => setBrowser('search')}>
              <FolderOpen size={15} />
              フォルダーを選択
            </button>
            <button
              className="secondary"
              disabled={!searchPath.trim() || saving || scanning || (config?.custom_paths.length || 0) >= 16}
              onClick={() => savePaths([...(config?.custom_paths || []), searchPath])}
            >
              探索先を追加
            </button>
          </div>
          <p className="subtle-note">
            追加した場所は保存されます。各探索先から4階層、最大2,000フォルダー・200候補を調べ、15秒を目安に区切ります。
          </p>
        </details>
        <div className="model-scan-status" role="status">
          {scanning ? (
            <>
              <Loader2 size={15} className="spin" />
              検出中 · {scan.scanned}フォルダーを確認
            </>
          ) : scan?.state === 'cancelled' ? (
            '検出を中止しました。見つかった候補を表示しています。'
          ) : scan?.state === 'failed' ? (
            scan.error || '検出に失敗しました。再検出してください。'
          ) : scan ? (
            `${scan.candidates.length}件の候補が見つかりました`
          ) : (
            '検出を準備しています…'
          )}
        </div>
        {scanError && (
          <p className="model-error" role="alert">
            {scanError}
          </p>
        )}
        {scan?.truncated && (
          <p className="subtle-note">
            探索上限に達しました。必要なモデルがない場合は、その近くのフォルダーを探索先に追加してください。
          </p>
        )}
        {!!scan?.warnings.length && (
          <details className="model-scan-warnings">
            <summary>読み取れなかった場所（{scan.warnings.length}件）</summary>
            {scan.warnings.map((w) => (
              <p key={w}>{w}</p>
            ))}
          </details>
        )}
        {!scanning && scan?.candidates.length === 0 && (
          <div className="model-discovery-empty">
            <FolderOpen size={26} />
            <p>モデルが見つかりませんでした。</p>
            <span>保存場所を上の欄に入力するか、探索先を追加してください。</span>
          </div>
        )}
        <div className="model-candidates">
          {scan?.candidates.map((candidate) => {
            const registered = models.find((m) => m.path === candidate.path);
            return (
              <article className="model-candidate" key={candidate.path}>
                <div className="model-candidate-icon">
                  <Cpu size={19} />
                </div>
                <div className="model-candidate-copy">
                  <div className="model-candidate-heading">
                    <h3>{candidate.name}</h3>
                    <span className={`tag ${candidate.status === 'available' ? 'good' : ''}`}>
                      {registered
                        ? '登録済み'
                        : candidate.status === 'available'
                          ? '登録可能'
                          : candidate.status === 'unsupported'
                            ? '非対応'
                            : 'ファイルを確認'}
                    </span>
                  </div>
                  <code>{candidate.path}</code>
                  <p>{candidate.reason}</p>
                  {candidate.weight_bytes > 0 && (
                    <small>
                      {candidate.architecture} · {bytes(candidate.weight_bytes)}
                    </small>
                  )}
                </div>
                {candidate.status === 'available' && (
                  <button
                    className="secondary"
                    disabled={busy}
                    onClick={() =>
                      registered
                        ? onRegistered(registered).catch((e) => onError(String(e)))
                        : choose(candidate)
                    }
                  >
                    {registered ? <Check size={15} /> : <ArrowUp size={15} />}
                    {registered ? '解析に使う' : 'この場所を指定'}
                  </button>
                )}
              </article>
            );
          })}
        </div>
      </section>

      <div className="model-inventory-heading">
        <h2>登録したモデル</h2>
        <button className="text-button" onClick={() => reload().catch((e) => onError(String(e)))}>
          <RefreshCw size={14} />
          状態を更新
        </button>
      </div>
      <div className="model-list">
        {models.map((m) => (
          <section key={m.id} className="panel model-card">
            <div className="model-card-top">
              <div className="model-icon">
                <Cpu />
              </div>
              <div>
                <h3>{m.name}</h3>
                <span className="mono muted">
                  {m.architecture} · {m.layers} blocks · d={m.hidden_size}
                </span>
              </div>
              <span className="tag">{m.trained ? '登録済み' : '検証用'}</span>
            </div>
            <div className="capabilities">
              {[
                'residual',
                'attention',
                'mlp',
                'generation',
                'quantization',
                'interventions',
                'tuned_lens',
              ].map((key) => (
                <div key={key}>
                  <span>{key}</span>
                  <b className={m.capabilities[key] === '検証済み' ? 'good' : 'muted'}>
                    {m.capabilities[key]}
                  </b>
                </div>
              ))}
            </div>
            {m.path && <p className="path-note">{m.path}</p>}
            {m.available === false && <p className="model-error">{m.availability_reason}</p>}
            {m.trained && (
              <div className="button-row">
                <button
                  className="secondary"
                  disabled={m.available === false || busy}
                  onClick={() => onRegistered(m).catch((e) => onError(String(e)))}
                >
                  解析に使う
                </button>
                <button
                  className="text-button"
                  disabled={busy}
                  onClick={() => {
                    setRelocating(m);
                    setPath(m.path || '');
                    setName(m.name);
                    input.current?.focus();
                  }}
                >
                  保存場所を再指定
                </button>
              </div>
            )}
          </section>
        ))}
      </div>
      {browser && (
        <FolderBrowser
          initialPath={browser === 'model' ? path : searchPath}
          environment={config?.backend_os || ''}
          onClose={() => setBrowser(undefined)}
          onChoose={(value) => {
            if (browser === 'model') setPath(value);
            else setSearchPath(value);
            setBrowser(undefined);
          }}
        />
      )}
    </div>
  );
}

type FolderListing = {
  path: string;
  parent: string;
  entries: { name: string; path: string }[];
  roots: { label: string; path: string }[];
  truncated: boolean;
};
function FolderBrowser({
  initialPath,
  environment,
  onClose,
  onChoose,
}: {
  initialPath: string;
  environment: string;
  onClose: () => void;
  onChoose: (path: string) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [location, setLocation] = useState(initialPath);
  const [destination, setDestination] = useState(initialPath);
  const [revision, setRevision] = useState(0);
  const [listing, setListing] = useState<FolderListing>();
  const [filter, setFilter] = useState('');
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState('');
  useEffect(() => {
    dialog.current?.showModal();
    return () => dialog.current?.close();
  }, []);
  useEffect(() => {
    let active = true;
    setBusy(true);
    setError('');
    setFilter('');
    api<FolderListing>(`models/folders?path=${encodeURIComponent(location)}`)
      .then((result) => {
        if (active) {
          setListing(result);
          setDestination(result.path);
        }
      })
      .catch((e) => {
        if (active) setError(String(e));
      })
      .finally(() => {
        if (active) setBusy(false);
      });
    return () => {
      active = false;
    };
  }, [location, revision]);
  return (
    <dialog
      ref={dialog}
      className="folder-browser"
      aria-labelledby="folder-browser-title"
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) {
          const r = e.currentTarget.getBoundingClientRect();
          if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)
            onClose();
        }
      }}
    >
      <div className="folder-browser-heading">
        <div>
          <small>{environment}の保存場所</small>
          <h2 id="folder-browser-title">フォルダーを選択</h2>
        </div>
        <button className="icon-button" onClick={onClose} aria-label="フォルダー選択を閉じる">
          <X size={19} />
        </button>
      </div>
      <form
        className="model-path-row"
        onSubmit={(e) => {
          e.preventDefault();
          setLocation(destination);
          setRevision((previous) => previous + 1);
        }}
      >
        <label className="visually-hidden" htmlFor="folder-browser-path">
          開くフォルダーの絶対パス
        </label>
        <input
          autoFocus
          id="folder-browser-path"
          disabled={busy}
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
        />
        <button className="secondary" disabled={busy}>
          移動
        </button>
      </form>
      <div className="folder-browser-tools">
        <button
          className="secondary"
          disabled={busy || !listing || listing.parent === listing.path}
          onClick={() => listing && setLocation(listing.parent)}
        >
          <ArrowUp size={15} />
          上の階層
        </button>
        <Select
          aria-label="保存場所のショートカット"
          value=""
          onChange={(e) => {
            if (e.target.value) setLocation(e.target.value === '__home__' ? '' : e.target.value);
          }}
        >
          <option value="">保存場所を選択</option>
          <option value="__home__">ホーム</option>
          {listing?.roots
            .filter((root) => root.label !== 'ホーム')
            .map((root, index) => (
              <option key={`${index}:${root.path}`} value={root.path}>
                {root.label}
              </option>
            ))}
        </Select>
      </div>
      <label className="field">
        フォルダー名で絞り込み
        <input value={filter} onChange={(e) => setFilter(e.target.value)} />
      </label>
      {error && (
        <p className="model-error" role="alert">
          {error}
        </p>
      )}
      <div className="folder-browser-list" aria-busy={busy}>
        {busy ? (
          <p role="status">
            <Loader2 size={18} className="spin" />
            フォルダーを確認中…
          </p>
        ) : (
          !error && (
            <>
              {listing?.entries
                .filter((entry) => entry.name.toLocaleLowerCase().includes(filter.toLocaleLowerCase()))
                .map((entry) => (
                  <button key={entry.path} onClick={() => setLocation(entry.path)}>
                    <FolderOpen size={17} />
                    <span>{entry.name}</span>
                    <ChevronRight size={15} />
                  </button>
                ))}
              {listing?.entries.length === 0 && <p>サブフォルダーはありません。</p>}
              {filter &&
                listing?.entries.length !== 0 &&
                !listing?.entries.some((entry) =>
                  entry.name.toLocaleLowerCase().includes(filter.toLocaleLowerCase()),
                ) && <p>一致するフォルダーはありません。</p>}
            </>
          )
        )}
      </div>
      {listing?.truncated && (
        <p className="subtle-note">表示上限300フォルダーに達しました。上のパス欄で直接移動できます。</p>
      )}
      <footer>
        <code>{listing?.path}</code>
        <div className="button-row">
          <button className="secondary" onClick={onClose}>
            キャンセル
          </button>
          <button
            className="primary"
            disabled={busy || !!error || !listing}
            onClick={() => listing && onChoose(listing.path)}
          >
            このフォルダーを使用
          </button>
        </div>
      </footer>
    </dialog>
  );
}
