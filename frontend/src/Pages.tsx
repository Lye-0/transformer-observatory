import { useEffect, useState } from 'react';
import {
  ArrowRight,
  Check,
  Cpu,
  Database,
  Download,
  FolderOpen,
  HardDrive,
  Layers3,
  Loader2,
  Plus,
  ShieldCheck,
  Trash2,
  Upload,
} from 'lucide-react';
import { api, bytes, upload, type Job, type Model } from './api';
import Select from './Select';
import { NumberField } from './NumberInput';

type Common = { onError: (message: string) => void };

export function NewAnalysis({
  models,
  onJob,
  onError,
  initialRequest,
  modelId,
  onModelChange: setModelId,
  onAddModel,
}: Common & {
  models: Model[];
  onJob: (job: Job) => void;
  initialRequest?: Record<string, any>;
  modelId: string;
  onModelChange: (id: string) => void;
  onAddModel: () => void;
}) {
  const [prompt, setPrompt] = useState('The quiet sky is full of possibilities.');
  const [title, setTitle] = useState('');
  const [level, setLevel] = useState(2);
  const [generate, setGenerate] = useState(false);
  const [maxNew, setMaxNew] = useState(16);
  const [device, setDevice] = useState('auto');
  const [dtype, setDtype] = useState('auto');
  const [layers, setLayers] = useState('');
  const [heads, setHeads] = useState('');
  const [tensorScope, setTensorScope] = useState('');
  const [attention, setAttention] = useState(false);
  const [seed, setSeed] = useState(42);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(1);
  const [projection, setProjection] = useState('auto');
  const [pf, setPf] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const model = models.find((m) => m.id === modelId);
  useEffect(() => {
    if (!initialRequest) return;
    const r = initialRequest;
    setModelId(r.model_id);
    setPrompt(r.prompt);
    setTitle(r.title || '');
    setLevel(r.level);
    setGenerate(r.generate || false);
    setMaxNew(r.max_new_tokens || 16);
    setDevice(r.device || 'auto');
    setDtype(r.dtype || 'auto');
    setLayers((r.layers || []).join(','));
    setHeads((r.heads || []).join(','));
    setTensorScope((r.tensors || []).join(','));
    setAttention(r.attention || false);
    setSeed(r.seed ?? 42);
    setTemperature(r.temperature ?? 0);
    setTopP(r.top_p ?? 1);
    setProjection(r.projection || 'auto');
  }, [initialRequest, setModelId]);
  const indices = (text: string) => (text.trim() ? text.split(',').map((x) => Number(x.trim())) : []);
  const payload = () => ({
    model_id: modelId,
    prompt,
    title,
    level,
    generate,
    max_new_tokens: maxNew,
    device,
    dtype,
    layers: indices(layers),
    heads: indices(heads),
    tensors: tensorScope
      .split(',')
      .map((x) => x.trim())
      .filter(Boolean),
    attention,
    seed,
    temperature,
    top_p: topP,
    projection,
  });
  useEffect(() => {
    setPf(null);
  }, [
    modelId,
    prompt,
    level,
    generate,
    maxNew,
    device,
    dtype,
    layers,
    heads,
    tensorScope,
    attention,
    seed,
    temperature,
    topP,
    projection,
  ]);
  async function check(start: boolean) {
    setBusy(true);
    try {
      const result = await api('analysis/preflight', 'POST', payload());
      setPf(result);
      if (start) onJob(await api<Job>('analysis/jobs', 'POST', payload()));
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="page analysis-page">
      <div className="eyebrow">
        NEW EXPERIMENT <span>01 / OBSERVE</span>
      </div>
      <h1>ことばの内側を、観測する。</h1>
      <p className="page-intro">入力から内部表現へ。各トークンの変化を、実際の演算データでたどります。</p>
      <div className="analysis-grid">
        <section className="panel input-panel">
          <div className="panel-heading">
            <span>
              <Layers3 size={17} /> 解析を設定
            </span>
            <span className="mono muted">LOCAL INFERENCE</span>
          </div>
          <div className="model-choice">
            <label className="field">
              モデル
              <Select aria-label="モデル" value={modelId} onChange={(e) => setModelId(e.target.value)}>
                {models.some((m) => m.trained) && (
                  <optgroup label="登録したモデル">
                    {models
                      .filter((m) => m.trained)
                      .map((m) => (
                        <option key={m.id} value={m.id} disabled={m.available === false}>
                          {m.name}
                          {m.available === false ? '（保存場所を確認）' : ''}
                        </option>
                      ))}
                  </optgroup>
                )}
                <optgroup label="検証用">
                  {models
                    .filter((m) => !m.trained)
                    .map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name}
                      </option>
                    ))}
                </optgroup>
              </Select>
            </label>
            <button className="secondary" onClick={onAddModel}>
              <Plus size={16} />
              モデルを追加
            </button>
          </div>
          {!models.some((m) => m.trained) && (
            <p className="model-empty">
              実モデルはまだ登録されていません。「モデルを追加」から検出・保存場所の指定ができます。
            </p>
          )}
          {model?.available === false && (
            <div className="model-unavailable" role="status">
              <p>{model.availability_reason}</p>
              <button className="text-button" onClick={onAddModel}>
                Model Managerで保存場所を再指定
              </button>
            </div>
          )}
          <div className="model-meta">
            <span>{model?.layers} blocks</span>
            <span>{model?.hidden_size} dimensions</span>
            <span>{model?.heads} query heads</span>
          </div>
          {model && !model.trained && (
            <p className="subtle-note">
              固定重みの数値検証モデルです。事前学習済みLLMの意味表現を示すものではありません。
            </p>
          )}
          <label className="field prompt-field">
            プロンプト
            <textarea
              aria-label="プロンプト"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={6}
              placeholder="観測したい文章を入力"
            />
          </label>
          <div className="field-row">
            <label className="field grow">
              実験名 <span className="muted">任意</span>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="プロンプトから自動で作成"
              />
            </label>
          </div>
          <fieldset className="levels">
            <legend>記録レベル</legend>
            <div className="level-options">
              {['Trajectory', 'Residual', 'Block', 'Deep Trace'].map((name, i) => (
                <button
                  type="button"
                  key={name}
                  className={level === i + 1 ? 'level-option selected' : 'level-option'}
                  aria-pressed={level === i + 1}
                  onClick={() => setLevel(i + 1)}
                >
                  <span className="mono">0{i + 1}</span>
                  <b>{name}</b>
                </button>
              ))}
            </div>
          </fieldset>
          <p className="subtle-note">
            {
              [
                '3D座標と高次元から算出した統計を保存します。',
                'Residualを保存し、再投影・プロンプト比較に利用できます。',
                'Residualと指定LayerのQ/K/V・MLPを保存します。',
                '取得可能な中間テンソルを広く保存します。対象範囲を指定できます。',
              ][level - 1]
            }
          </p>
          <div className="generation-row">
            <label className="check-field">
              <input type="checkbox" checked={generate} onChange={(e) => setGenerate(e.target.checked)} />
              続きを生成する
            </label>
            {generate && (
              <NumberField
                label="最大生成数"
                className="field inline-field"
                min={1}
                max={256}
                value={maxNew}
                onValueChange={(value) => setMaxNew(Number(value))}
              />
            )}
          </div>
          <details className="advanced">
            <summary>取得範囲・実行条件</summary>
            <div className="advanced-body">
              <div className="field-row">
                <label className="field">
                  デバイス
                  <Select aria-label="デバイス" value={device} onChange={(e) => setDevice(e.target.value)}>
                    <option value="auto">自動</option>
                    <option value="cuda">CUDA</option>
                    <option value="cpu">CPU</option>
                  </Select>
                </label>
                <label className="field">
                  計算精度
                  <Select aria-label="計算精度" value={dtype} onChange={(e) => setDtype(e.target.value)}>
                    {['auto', 'float32', 'float16', 'bfloat16'].map((v) => (
                      <option key={v}>{v}</option>
                    ))}
                  </Select>
                </label>
              </div>
              {level >= 3 && (
                <>
                  <div className="field-row">
                    <label className="field">
                      Layer番号
                      <input
                        placeholder="空欄で全層 · 例 0,1,2"
                        value={layers}
                        onChange={(e) => setLayers(e.target.value)}
                      />
                    </label>
                    <label className="field">
                      Query Head番号
                      <input
                        placeholder="空欄で全Head · 例 0,1"
                        value={heads}
                        onChange={(e) => setHeads(e.target.value)}
                      />
                    </label>
                  </div>
                  <label className="field">
                    テンソル名（空欄で取得可能な全種類）
                    <input
                      value={tensorScope}
                      onChange={(e) => setTensorScope(e.target.value)}
                      placeholder="例 q,k,v,attention_weights,gate,up,product"
                    />
                  </label>
                  <label className="check-field">
                    <input
                      type="checkbox"
                      checked={attention}
                      onChange={(e) => setAttention(e.target.checked)}
                    />
                    Attention係数行列も保存する
                  </label>
                </>
              )}
              <div className="field-row">
                <NumberField
                  label="Seed"
                  min={0}
                  value={seed}
                  onValueChange={(value) => setSeed(Number(value))}
                />
                <label className="field">
                  PCA
                  <Select
                    aria-label="PCA方式"
                    value={projection}
                    onChange={(e) => setProjection(e.target.value)}
                  >
                    <option value="auto">自動</option>
                    <option value="exact">Exact</option>
                    <option value="incremental">Incremental</option>
                  </Select>
                </label>
              </div>
              {generate && (
                <div className="field-row">
                  <NumberField
                    label="Temperature（0でGreedy）"
                    min={0}
                    max={2}
                    step="0.1"
                    value={temperature}
                    onValueChange={(value) => setTemperature(Number(value))}
                  />
                  <NumberField
                    label="Top-p"
                    min="0.01"
                    max={1}
                    step="0.05"
                    value={topP}
                    onValueChange={(value) => setTopP(Number(value))}
                  />
                </div>
              )}
            </div>
          </details>
          <div className="form-actions">
            <button
              className="secondary"
              disabled={busy || !prompt || !model || model.available === false}
              onClick={() => check(false)}
            >
              トークン・容量を確認
            </button>
            <button
              className="primary"
              disabled={busy || !prompt || !model || model.available === false}
              onClick={() => check(true)}
            >
              {busy ? <Loader2 className="spin" size={17} /> : <ArrowRight size={17} />}解析を開始
            </button>
          </div>
        </section>
        <aside className="preflight-column">
          <section className="panel preflight-panel">
            <div className="panel-heading">
              <span>
                <ShieldCheck size={17} /> 実行前チェック
              </span>
            </div>
            <div className="preflight-orbit">
              <div />
              <div />
              <span>∴</span>
            </div>
            <h3>{pf ? '解析の準備ができています' : '実測のための準備'}</h3>
            <p className="muted">Tokenizer・取得範囲・利用可能なリソースを確認してから実行します。</p>
            <dl className="resource-list">
              <div>
                <dt>
                  <Layers3 size={15} />
                  入力トークン
                </dt>
                <dd>{pf?.input_tokens ?? '—'}</dd>
              </div>
              <div>
                <dt>
                  <Cpu size={15} />
                  実行環境
                </dt>
                <dd>{pf ? `${pf.device.toUpperCase()} · ${pf.dtype}` : '自動選択'}</dd>
              </div>
              <div>
                <dt>
                  <Cpu size={15} />
                  RAMの目安
                </dt>
                <dd>
                  {pf ? `${bytes(pf.estimated_ram_bytes)} / 空き ${bytes(pf.ram_available_bytes)}` : '—'}
                </dd>
              </div>
              <div>
                <dt>
                  <Cpu size={15} />
                  VRAMの目安
                </dt>
                <dd>
                  {pf
                    ? pf.device === 'cuda'
                      ? `${bytes(pf.estimated_vram_bytes)} / 空き ${bytes(pf.gpu_free_bytes)}`
                      : 'CPUを使用'
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>
                  <HardDrive size={15} />
                  一時容量の目安
                </dt>
                <dd>{pf ? bytes(pf.estimated_spool_bytes) : '—'}</dd>
              </div>
              <div>
                <dt>
                  <Database size={15} />
                  最終保存の目安
                </dt>
                <dd>{pf ? bytes(pf.estimated_final_bytes) : '—'}</dd>
              </div>
            </dl>
            {pf?.warnings.map((w: string) => (
              <p key={w} className="subtle-note">
                {w}
              </p>
            ))}
          </section>
          <section className="principle">
            <span className="eyebrow">OBSERVATION PRINCIPLE</span>
            <h3>すべての軌跡を、そのままに。</h3>
            <p>全トークン位置の軌跡を描画。高次元の実測値と3Dへの射影を区別して観測します。</p>
          </section>
        </aside>
      </div>
      {pf && (
        <section className="panel token-preview">
          <div className="panel-heading">
            <span>実際のTokenizerによる分割</span>
            <span>{pf.tokens.length} TOKENS</span>
          </div>
          <div className="token-chips">
            {pf.tokens.map((t: any) => (
              <span key={t.position} title={`Position ${t.position} / ID ${t.id}`}>
                <small>{t.position}</small>
                {t.raw}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export { default as ModelManager } from './ModelManager';

export function Library({ onOpen, onError }: Common & { onOpen: (id: string) => void }) {
  const [traces, setTraces] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<any>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [partial, setPartial] = useState<any>(null);
  const [editing, setEditing] = useState<any>(null);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('date');
  const refresh = () =>
    api<any[]>('traces')
      .then((items) => {
        setTraces(items);
        setSelected((old) => old.filter((id) => items.some((t) => t.id === id && !t.retained && !t.active)));
      })
      .catch((e) => onError(String(e)));
  useEffect(() => {
    refresh();
  }, []);
  async function action(task: () => Promise<unknown>) {
    setBusy(true);
    try {
      await task();
    } catch (e) {
      onError(String(e));
    } finally {
      await refresh();
      setBusy(false);
    }
  }
  return (
    <div className="page">
      <div className="eyebrow">TRACE LIBRARY</div>
      <div className="page-title-row">
        <div>
          <h1>観測の記録。</h1>
          <p className="page-intro">保存した実験は、モデルの再実行なしで閲覧できます。</p>
        </div>
        <div className="button-row">
          {selected.length > 0 && (
            <button
              className="danger"
              disabled={busy}
              onClick={() => {
                if (confirm(`選択した${selected.length}件の一時トレースを削除します。よろしいですか？`))
                  action(async () => {
                    for (const id of selected)
                      await api(`traces/${id}`, 'DELETE', undefined, { 'X-Confirm-Trace': id });
                    setSelected([]);
                  });
              }}
            >
              選択した{selected.length}件を削除
            </button>
          )}
          <label className="button secondary">
            <Upload size={16} />
            インポート
            <input
              className="visually-hidden"
              type="file"
              accept=".zip"
              disabled={busy}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) action(() => upload(f));
                e.target.value = '';
              }}
            />
          </label>
          <button
            className="secondary"
            disabled={busy}
            onClick={() => action(async () => setPreview(await api('storage/cleanup/preview', 'POST')))}
          >
            <Trash2 size={16} />
            一時データ整理
          </button>
        </div>
      </div>
      <div className="field-row">
        <label className="field grow">
          実験を検索
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="名前・モデル・説明・タグ"
          />
        </label>
        <label className="field">
          並び順
          <Select aria-label="実験の並び順" value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="date">新しい順</option>
            <option value="size">容量が大きい順</option>
          </Select>
        </label>
      </div>
      {editing && (
        <section className="panel cleanup-preview" role="region" aria-label="実験の情報を編集">
          <label className="field">
            実験名
            <input
              value={editing.title}
              maxLength={160}
              onChange={(e) => setEditing({ ...editing, title: e.target.value })}
            />
          </label>
          <label className="field">
            説明
            <textarea
              value={editing.description || ''}
              maxLength={4000}
              onChange={(e) => setEditing({ ...editing, description: e.target.value })}
            />
          </label>
          <label className="field">
            タグ（カンマ区切り）
            <input
              value={editing.tagsText}
              onChange={(e) => setEditing({ ...editing, tagsText: e.target.value })}
            />
          </label>
          <div className="button-row">
            <button className="secondary" onClick={() => setEditing(null)}>
              閉じる
            </button>
            <button
              className="primary"
              disabled={busy || !editing.title.trim()}
              onClick={() =>
                action(async () => {
                  await api(`traces/${editing.id}`, 'PATCH', {
                    title: editing.title,
                    description: editing.description || '',
                    tags: editing.tagsText
                      .split(',')
                      .map((t: string) => t.trim())
                      .filter(Boolean),
                  });
                  setEditing(null);
                })
              }
            >
              情報を保存
            </button>
          </div>
        </section>
      )}
      {partial && (
        <section className="panel cleanup-preview" role="region" aria-label="部分トレースの概要">
          <div className="panel-heading">
            <b>{partial.title} · 部分トレース</b>
            <button className="secondary compact" onClick={() => setPartial(null)}>
              閉じる
            </button>
          </div>
          <p className="warning">{partial.error || '取得中または未完了のデータです'}</p>
          <p>
            {partial.stages.length}種類の観測点 · {partial.tensors.length}
            チャンク。未完了の3D軌跡は作成しません。
          </p>
          <p className="path-note">
            {partial.storage_root} / {partial.id}
          </p>
          <details>
            <summary>取得できたチャンク</summary>
            <div className="difference-table">
              <table>
                <thead>
                  <tr>
                    <th>段階</th>
                    <th>生成Step</th>
                    <th>形状</th>
                    <th>ファイル</th>
                  </tr>
                </thead>
                <tbody>
                  {partial.tensors.map((r: any) => (
                    <tr key={r.file}>
                      <td>{r.stage}</td>
                      <td>{r.step}</td>
                      <td>{r.shape.join(' × ')}</td>
                      <td>{r.file}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      )}
      {preview && (
        <section className="panel cleanup-preview" role="region" aria-label="削除プレビュー">
          <h3>
            整理対象：{preview.targets.length}件 · {bytes(preview.bytes)}
          </h3>
          <p>保持中・実行中のトレースは除外されています。</p>
          <ul>
            {preview.targets.map((t: any) => (
              <li key={t.id}>
                {t.title} · {bytes(t.bytes)}
              </li>
            ))}
          </ul>
          <div className="button-row">
            <button className="secondary" onClick={() => setPreview(null)}>
              閉じる
            </button>
            <button
              className="danger"
              disabled={!preview.targets.length || busy}
              onClick={() => {
                if (confirm(`${preview.targets.length}件の一時トレースを削除します。よろしいですか？`))
                  action(async () => {
                    await api('storage/cleanup/confirm', 'POST', { token: preview.token });
                    setPreview(null);
                  });
              }}
            >
              対象を削除
            </button>
          </div>
        </section>
      )}
      {!traces.length ? (
        <div className="empty-state">
          <Database size={36} />
          <h3>まだ観測の記録がありません</h3>
          <p>「新しい解析」から最初の実験を始めてください。</p>
        </div>
      ) : (
        <div className="trace-list">
          {traces
            .filter((t) =>
              `${t.title} ${t.model} ${t.description} ${(t.tags || []).join(' ')}`
                .toLowerCase()
                .includes(query.toLowerCase()),
            )
            .sort((a, b) => (sort === 'size' ? b.size_bytes - a.size_bytes : b.created_at - a.created_at))
            .map((t) => (
              <article className="panel trace-row" key={t.id}>
                <input
                  type="checkbox"
                  aria-label={`削除対象に選択：${t.title}`}
                  disabled={busy || t.retained || t.active}
                  checked={selected.includes(t.id)}
                  onChange={(e) =>
                    setSelected((items) =>
                      e.target.checked ? [...items, t.id] : items.filter((id) => id !== t.id),
                    )
                  }
                />
                <button
                  className="trace-open"
                  disabled={t.status !== 'completed'}
                  onClick={() => onOpen(t.id)}
                >
                  <span className="trace-symbol">
                    <Layers3 size={21} />
                  </span>
                  <span>
                    <b>{t.title}</b>
                    <small>
                      {t.model} · {t.token_count} tokens · Level {t.level}
                    </small>
                  </span>
                </button>
                <div className="trace-state">
                  <span className={t.status === 'completed' ? 'tag good' : 'tag warning'}>
                    {t.status === 'completed' ? '完了' : '不完全'}
                  </span>
                  <small>
                    {new Date(t.created_at * 1000).toLocaleString('ja-JP')} · {bytes(t.size_bytes)}
                  </small>
                  {t.error && <small className="warning">{t.error}</small>}
                  {t.tags?.length > 0 && <small>{t.tags.join(' · ')}</small>}
                  {t.status !== 'completed' && (
                    <button
                      className="text-button"
                      onClick={() => action(async () => setPartial(await api(`traces/${t.id}`)))}
                    >
                      部分データを確認
                    </button>
                  )}
                </div>
                <div className="button-row">
                  <button
                    className="text-button"
                    disabled={busy || t.active}
                    onClick={() => setEditing({ ...t, tagsText: (t.tags || []).join(', ') })}
                  >
                    編集
                  </button>
                  <button
                    className="icon-button"
                    title="保存先を開く"
                    disabled={busy}
                    onClick={() => action(() => api('storage/open', 'POST', { trace_id: t.id }))}
                  >
                    <FolderOpen size={18} />
                  </button>
                  <button
                    title={t.retained ? '保持を解除' : 'この実験を保持'}
                    className={t.retained ? 'icon-button active' : 'icon-button'}
                    disabled={busy || t.active}
                    onClick={() => action(() => api(`traces/${t.id}/retain?enabled=${!t.retained}`, 'POST'))}
                  >
                    <ShieldCheck size={18} />
                  </button>
                  {t.status === 'completed' && (
                    <a className="icon-button" title="エクスポート" href={`/api/traces/${t.id}/export`}>
                      <Download size={18} />
                    </a>
                  )}
                  <button
                    className="icon-button danger-text"
                    title="削除"
                    disabled={busy || t.retained || t.active}
                    onClick={() => {
                      if (confirm(`「${t.title}」を削除します。よろしいですか？`))
                        action(() => api(`traces/${t.id}`, 'DELETE', undefined, { 'X-Confirm-Trace': t.id }));
                    }}
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              </article>
            ))}
        </div>
      )}
    </div>
  );
}

export function Settings({ onError }: Common) {
  const [settings, setSettings] = useState<any>(null);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    api('settings')
      .then(setSettings)
      .catch((e) => onError(String(e)));
  }, []);
  if (!settings) return <div className="page">設定を読み込み中…</div>;
  const change = (key: string, value: string) => {
    setSaved(false);
    setSettings({ ...settings, [key]: Number(value) });
  };
  async function save() {
    try {
      const { retention_days, cache_limit_gib, trace_limit_gib, ram_budget_mib, trace_root } = settings;
      setSettings(
        await api('settings', 'PATCH', {
          retention_days,
          cache_limit_gib,
          trace_limit_gib,
          ram_budget_mib,
          trace_root,
        }),
      );
      setSaved(true);
    } catch (e) {
      onError(String(e));
    }
  }
  return (
    <div className="page settings-page">
      <div className="eyebrow">SETTINGS</div>
      <h1>研究環境を整える。</h1>
      <p className="page-intro">設定は保存後、新しい解析から適用されます。</p>
      <section className="panel">
        <div className="panel-heading">
          <span>
            <HardDrive size={17} /> 保存とリソース
          </span>
          <span className="tag">{settings.backend_os}</span>
        </div>
        <div className="field-row">
          <NumberField
            label="一時保持期間（日）"
            min={1}
            max={3650}
            value={settings.retention_days}
            onValueChange={(value) => change('retention_days', value)}
          />
          <NumberField
            label="一時データ上限（GiB）"
            min="0.1"
            step="0.1"
            value={settings.cache_limit_gib}
            onValueChange={(value) => change('cache_limit_gib', value)}
          />
        </div>
        <div className="field-row">
          <NumberField
            label="1トレースの上限（GiB）"
            min="0.05"
            step="0.05"
            value={settings.trace_limit_gib}
            onValueChange={(value) => change('trace_limit_gib', value)}
          />
          <NumberField
            label="PCA用RAM予算（MiB）"
            min={64}
            value={settings.ram_budget_mib}
            onValueChange={(value) => change('ram_budget_mib', value)}
          />
        </div>
        <label className="field">
          新しいトレースの保存先（バックエンドOSの絶対パス）
          <input
            value={settings.trace_root}
            onChange={(e) => {
              setSaved(false);
              setSettings({ ...settings, trace_root: e.target.value });
            }}
          />
        </label>
        <dl className="resource-list">
          <div>
            <dt>一時トレース</dt>
            <dd className="path-note">{settings.trace_root}</dd>
          </div>
          <div>
            <dt>設定・実験一覧</dt>
            <dd className="path-note">{settings.data_root}</dd>
          </div>
          <div>
            <dt>ディスク空き</dt>
            <dd>{bytes(settings.free_disk_bytes)}</dd>
          </div>
        </dl>
        <p className="subtle-note">
          保存すると新しい解析から適用されます。既存トレースは元の場所から閲覧できます。保持した実験は、整合性確認後に永続領域へ移します。起動時の環境変数を指定した場合は、その保存先が優先されます。
        </p>
        <div className="form-actions">
          <button
            className="secondary"
            onClick={async () => {
              try {
                await api('storage/open', 'POST', {});
              } catch (e) {
                onError(String(e));
              }
            }}
          >
            <FolderOpen size={16} />
            現在の保存先を開く
          </button>
          {saved && (
            <span className="good">
              <Check size={15} /> 保存しました
            </span>
          )}
          <button className="primary" onClick={save}>
            設定を保存
          </button>
        </div>
      </section>
    </div>
  );
}
