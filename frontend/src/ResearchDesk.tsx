import { useEffect, useState } from 'react';
import { Activity, Download, Loader2 } from 'lucide-react';
import { api, number, type Manifest, type Model } from './api';
import Select from './Select';
import TrajectoryScene from './TrajectoryScene';

type Props = { models: Model[]; onError: (message: string) => void };
const labels: Record<string, string> = {
  intervention: '介入実験',
  statistics: '複数実験の統計',
  tsne: '追加射影',
  lenses: 'Tuned Lens',
};
const terminal = ['completed', 'failed', 'cancelled'];

export default function ResearchDesk({ models, onError }: Props) {
  const [tab, setTab] = useState('intervention');
  const [traces, setTraces] = useState<any[]>([]);
  const [target, setTarget] = useState('');
  const [source, setSource] = useState('');
  const [manifest, setManifest] = useState<Manifest>();
  const [sourceManifest, setSourceManifest] = useState<Manifest>();
  const [mode, setMode] = useState('residual_patch');
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [stage, setStage] = useState('embedding');
  const [positions, setPositions] = useState('');
  const [sourcePositions, setSourcePositions] = useState('');
  const [readout, setReadout] = useState('');
  const [tokenId, setTokenId] = useState('');
  const [strength, setStrength] = useState(1);
  const [statisticsIds, setStatisticsIds] = useState<string[]>([]);
  const [perplexity, setPerplexity] = useState(20);
  const [seed, setSeed] = useState(42);
  const [history, setHistory] = useState<any[]>([]);
  const [studyId, setStudyId] = useState('');
  const [study, setStudy] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [lenses, setLenses] = useState<any[]>([]);
  const [lensPath, setLensPath] = useState('');
  const [lensModel, setLensModel] = useState(models[0]?.id || '');

  async function refresh() {
    const items = await api<any[]>('research/studies');
    setHistory(items);
    setStudyId((old) => old || items[0]?.id || '');
  }
  useEffect(() => {
    let active = true;
    Promise.all([api<any[]>('traces'), api<any[]>('research/lenses'), api<any[]>('research/studies')])
      .then(([traces, lenses, studies]) => {
        if (!active) return;
        const ready = traces.filter((t) => t.status === 'completed');
        const compatible = ready.filter((t) => t.model_fingerprint === ready[0]?.model_fingerprint);
        const groups = new Map<string, any[]>();
        for (const trace of ready)
          groups.set(trace.model_fingerprint, [...(groups.get(trace.model_fingerprint) || []), trace]);
        const statisticsGroup = [...groups.values()].find((group) => group.length >= 2) || [];
        setTraces(ready);
        setTarget(ready[0]?.id || '');
        setSource(compatible[1]?.id || compatible[0]?.id || '');
        setStatisticsIds(statisticsGroup.slice(0, 2).map((t) => t.id));
        setLenses(lenses);
        setHistory(studies);
        setStudyId(studies[0]?.id || '');
      })
      .catch((e) => onError(String(e)));
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    let active = true;
    setManifest(undefined);
    if (target)
      api<Manifest>(`traces/${target}`)
        .then((m) => {
          if (active) {
            setManifest(m);
            setLayer(0);
            setHead(0);
            setStage('embedding');
            setSource((previous) =>
              traces.find((t) => t.id === previous)?.model_fingerprint === m.model.fingerprint
                ? previous
                : traces.find((t) => t.model_fingerprint === m.model.fingerprint && t.id !== m.id)?.id ||
                  m.id,
            );
          }
        })
        .catch((e) => onError(String(e)));
    return () => {
      active = false;
    };
  }, [target]);
  useEffect(() => {
    let active = true;
    setSourceManifest(undefined);
    if (source)
      api<Manifest>(`traces/${source}`)
        .then((m) => {
          if (active) setSourceManifest(m);
        })
        .catch((e) => onError(String(e)));
    return () => {
      active = false;
    };
  }, [source]);
  useEffect(() => {
    if (!studyId) return;
    let active = true,
      timer: number | undefined;
    const poll = async () => {
      try {
        const item = await api(`research/studies/${studyId}`);
        if (!active) return;
        setStudy(item);
        if (!terminal.includes(item.state)) timer = window.setTimeout(poll, 1500);
        else refresh().catch((e) => onError(String(e)));
      } catch (e) {
        if (active) onError(String(e));
      }
    };
    setStudy(null);
    poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [studyId]);
  const indices = (text: string) => {
    if (!text.trim()) return [];
    const parts = text.split(',');
    if (parts.some((s) => !s.trim() || !Number.isInteger(Number(s)) || Number(s) < 0))
      throw new Error('位置は0以上の整数をカンマで区切って指定してください');
    return parts.map(Number);
  };
  async function submit() {
    setBusy(true);
    try {
      const result = await api('research/studies', 'POST', {
        kind: tab,
        trace_ids:
          tab === 'statistics'
            ? statisticsIds
            : tab === 'intervention' && mode === 'residual_patch'
              ? [target, source]
              : [target],
        mode,
        layer,
        head,
        stage,
        positions: indices(positions),
        source_positions: indices(sourcePositions),
        readout_position: readout.trim() ? Number(readout) : null,
        token_id: tokenId.trim() ? Number(tokenId) : null,
        strength,
        perplexity,
        seed,
      });
      setStudyId(result.id);
      await refresh();
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function registerLens() {
    setBusy(true);
    try {
      await api('research/lenses/register', 'POST', { model_id: lensModel, path: lensPath });
      setLenses(await api('research/lenses'));
      setLensPath('');
    } catch (e) {
      onError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const selector = (value: string, change: (id: string) => void, label: string, fingerprint?: string) => (
    <label className="field grow">
      {label}
      <Select aria-label={label} value={value} onChange={(e) => change(e.target.value)}>
        <option value="">実験を選択</option>
        {traces.map((t) => (
          <option key={t.id} value={t.id} disabled={!!fingerprint && t.model_fingerprint !== fingerprint}>
            {t.title} · Level {t.level}
          </option>
        ))}
      </Select>
    </label>
  );
  const inputTokens = manifest?.tokens.filter((t) => !t.generated) || [];
  const sourceTokens = sourceManifest?.tokens || [];
  return (
    <div className="page research-page">
      <div className="eyebrow">RESEARCH WORKBENCH</div>
      <h1>観察から、条件を変えた実験へ。</h1>
      <p className="page-intro">
        元のトレースは維持し、介入・記述統計・非線形射影を別の研究レポートへ記録します。
      </p>
      <div className="research-tabs segmented" aria-label="研究機能">
        {Object.entries(labels).map(([key, label]) => (
          <button key={key} className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>
            {label}
          </button>
        ))}
      </div>
      <section className="panel research-form">
        {tab === 'intervention' && (
          <>
            <div className="panel-heading">
              <span>
                <Activity size={17} />
                指定位置への介入
              </span>
              <span className="tag">PREFILL</span>
            </div>
            <p className="subtle-note">
              同じモデルで基準・介入あり・介入解除後を実際に再計算します。読み出しは元入力の指定位置の次トークン分布です。生成済み位置への介入は現在対象外です。
            </p>
            <div className="field-row">
              {selector(target, setTarget, '介入先の実験')}
              <label className="field">
                介入方式
                <Select aria-label="介入方式" value={mode} onChange={(e) => setMode(e.target.value)}>
                  <option value="residual_patch">Residual Patch</option>
                  <option value="head_ablation">Head Ablation</option>
                  <option value="mlp_ablation">MLP Ablation</option>
                </Select>
              </label>
            </div>
            {mode === 'residual_patch' ? (
              <>
                <div className="field-row">
                  {selector(source, setSource, 'Patch取得元の実験', manifest?.model.fingerprint)}
                  <label className="field">
                    Patchする観測点
                    <Select
                      aria-label="Patchする観測点"
                      value={stage}
                      onChange={(e) => setStage(e.target.value)}
                    >
                      {manifest?.stages
                        .filter((s) => s === 'embedding' || s.endsWith('.mlp'))
                        .map((s) => (
                          <option key={s} value={s}>
                            {s}
                          </option>
                        ))}
                    </Select>
                  </label>
                </div>
                <label className="field">
                  取得元の位置（カンマ区切り）
                  <input
                    value={sourcePositions}
                    onChange={(e) => setSourcePositions(e.target.value)}
                    placeholder="空欄＝取得元入力の末尾"
                  />
                </label>
                <details>
                  <summary>取得元の実トークン位置</summary>
                  <div className="token-chips">
                    {sourceTokens.map((t) => (
                      <span key={t.position} title={`ID ${t.id}`}>
                        <small>{t.position}</small>
                        {t.raw}
                        {t.generated ? ' / Decode' : ''}
                      </span>
                    ))}
                  </div>
                </details>
                <p className="subtle-note">
                  取得元と介入先は同じ重み・Tokenizerが必要です。EmbeddingまたはBlock出力を置き換えます。Attention後のResidual
                  Patchは対応範囲に含めません。
                </p>
              </>
            ) : (
              <div className="field-row">
                <label className="field">
                  介入Layer
                  <input
                    type="number"
                    min={0}
                    max={(manifest?.model.layers || 1) - 1}
                    value={layer}
                    onChange={(e) => setLayer(Number(e.target.value))}
                  />
                </label>
                {mode === 'head_ablation' && (
                  <label className="field">
                    無効化するQuery Head
                    <input
                      type="number"
                      min={0}
                      max={(manifest?.model.heads || 1) - 1}
                      value={head}
                      onChange={(e) => setHead(Number(e.target.value))}
                    />
                  </label>
                )}
              </div>
            )}
            <div className="field-row">
              <label className="field grow">
                介入先の位置（カンマ区切り）
                <input
                  value={positions}
                  onChange={(e) => setPositions(e.target.value)}
                  placeholder="空欄＝入力の末尾"
                />
              </label>
              <label className="field">
                介入の強さ
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.1}
                  value={strength}
                  onChange={(e) => setStrength(Number(e.target.value))}
                />
              </label>
            </div>
            <div className="field-row">
              <label className="field">
                出力を読む位置
                <input
                  type="number"
                  min={0}
                  value={readout}
                  onChange={(e) => setReadout(e.target.value)}
                  placeholder="空欄＝入力の末尾"
                />
              </label>
              <label className="field">
                注目する出力Token ID（任意）
                <input
                  type="number"
                  min={0}
                  value={tokenId}
                  onChange={(e) => setTokenId(e.target.value)}
                  placeholder="上位候補に加えて調べるID"
                />
              </label>
            </div>
            <details>
              <summary>介入先の実入力トークン</summary>
              <div className="token-chips">
                {inputTokens.map((t) => (
                  <span key={t.position} title={`ID ${t.id}`}>
                    <small>{t.position}</small>
                    {t.raw}
                  </span>
                ))}
              </div>
            </details>
            <p className="subtle-note">
              HeadはW_O直前の対応ブロック、MLPはResidualへ加算する出力を縮小します。強さ1で完全無効化、0は介入なしの対照です。モデル重みは書き換えません。
            </p>
          </>
        )}
        {tab === 'statistics' && (
          <>
            <h3>複数実験の記述統計</h3>
            <p className="subtle-note">
              同じモデルの実験を2〜20件選択します。各実験内で有効トークンの指標を平均し、その実験平均を等しい重みで集計します。独立PCAに依存する3D更新距離は除外し、統計的有意差や因果性を主張しません。
            </p>
            <div className="research-checks">
              {traces.map((t) => (
                <label className="check-field" key={t.id}>
                  <input
                    type="checkbox"
                    checked={statisticsIds.includes(t.id)}
                    disabled={
                      statisticsIds.length > 0 &&
                      t.model_fingerprint !==
                        traces.find((item) => item.id === statisticsIds[0])?.model_fingerprint
                    }
                    onChange={(e) =>
                      setStatisticsIds((old) =>
                        e.target.checked ? [...old, t.id] : old.filter((id) => id !== t.id),
                      )
                    }
                  />
                  {t.title}
                </label>
              ))}
            </div>
          </>
        )}
        {tab === 'tsne' && (
          <>
            <h3>全観測点のt-SNE</h3>
            {selector(target, setTarget, '追加射影する実験')}
            <p className="subtle-note">
              近傍構造を探索する非線形の別表示です。点間の距離や線の長さを、元の高次元距離・変位量として解釈しないでください。Scientific
              /
              Displacementの固定PCAは維持します。全点を使用し、4096観測点・RAM予算を超えたら抽出せず停止します。
            </p>
            <div className="field-row">
              <label className="field">
                Perplexity
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={perplexity}
                  onChange={(e) => setPerplexity(Number(e.target.value))}
                />
              </label>
              <label className="field">
                射影Seed
                <input type="number" min={0} value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              </label>
            </div>
          </>
        )}
        {tab === 'lenses' && (
          <>
            <h3>学習済みTuned Lensを登録</h3>
            <p className="subtle-note">
              公式linear_tuned_lensのconfig.jsonとparams.safetensors／params.ptを含むローカルフォルダーを登録します。未学習の変換を自動生成しません。登録後、ExplorerのPredictionから選択できます。
            </p>
            <label className="field">
              Lensの対象モデル
              <Select
                aria-label="Lensの対象モデル"
                value={lensModel}
                onChange={(e) => setLensModel(e.target.value)}
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </Select>
            </label>
            <label className="field">
              学習済みLensフォルダー
              <input
                value={lensPath}
                onChange={(e) => setLensPath(e.target.value)}
                placeholder="バックエンドOS上の絶対パス"
              />
            </label>
            <p className="subtle-note">
              重みと対象モデルの一致を記録するため、config.jsonのobservatory_model_fingerprintに学習対象の値が必要です。名前だけで重みを関連付けません。
            </p>
            <code className="fingerprint">{models.find((m) => m.id === lensModel)?.fingerprint}</code>
            <button className="primary" disabled={busy || !lensPath} onClick={registerLens}>
              学習済みLensを登録
            </button>
            {lenses.map((l) => (
              <article className="panel" key={l.id}>
                <h3>{l.name}</h3>
                <p className="path-note">{l.path}</p>
                <p className="subtle-note">提供された学習情報：{l.training_info}</p>
              </article>
            ))}
            {!lenses.length && (
              <p className="subtle-note">
                学習済み重みはまだ登録されていません。通常のLogit Lensは使用できます。
              </p>
            )}
          </>
        )}
        {tab !== 'lenses' && (
          <div className="form-actions">
            <button
              className="primary"
              disabled={busy || !target || (tab === 'statistics' && statisticsIds.length < 2)}
              onClick={submit}
            >
              {busy ? <Loader2 className="spin" size={17} /> : <Activity size={17} />}研究ジョブを開始
            </button>
          </div>
        )}
      </section>
      <section className="panel research-results">
        <div className="panel-heading">
          <span>研究レポート</span>
          <Select aria-label="研究レポート" value={studyId} onChange={(e) => setStudyId(e.target.value)}>
            <option value="">レポートを選択</option>
            {history.map((s) => (
              <option key={s.id} value={s.id}>
                {labels[s.kind]} · {new Date(s.created_at * 1000).toLocaleString('ja-JP')} · {s.state}
              </option>
            ))}
          </Select>
        </div>
        {study && (
          <>
            <p aria-live="polite">{study.phase}</p>
            {study.error && <p className="warning">{study.error}</p>}
            {!terminal.includes(study.state) && (
              <button
                className="secondary"
                onClick={async () => {
                  try {
                    setStudy(await api(`research/studies/${study.id}/cancel`, 'POST'));
                  } catch (e) {
                    onError(String(e));
                  }
                }}
              >
                研究ジョブを中止
              </button>
            )}
            {study.report && (
              <>
                <ReportConditions report={study.report} />
                <div className="button-row">
                  <a
                    className="button secondary"
                    href={`/api/research/studies/${study.id}/export?format=json`}
                  >
                    <Download size={15} />
                    条件付きJSON
                  </a>
                  <a
                    className="button secondary"
                    href={`/api/research/studies/${study.id}/export?format=csv`}
                  >
                    <Download size={15} />
                    CSV
                  </a>
                </div>
                <p className="subtle-note">{study.report.result.interpretation}</p>
                <StudyResult report={study.report} />
              </>
            )}
            {study.diagnostic && (
              <details>
                <summary>診断情報</summary>
                <pre className="tensor-numbers">{study.diagnostic}</pre>
              </details>
            )}
          </>
        )}
        {!studyId && (
          <p className="muted">
            研究ジョブを実行すると、条件と結果がここに保存されます。ブラウザを閉じても、バックエンド内で継続します。
          </p>
        )}
      </section>
    </div>
  );
}

function ReportConditions({ report }: { report: any }) {
  const request = report.request;
  const result = report.result;
  return (
    <details className="research-conditions" open>
      <summary>このレポートの実行条件</summary>
      <dl className="resource-list">
        {report.sources.map((source: any, i: number) => (
          <div key={`${i}:${source.id}`}>
            <dt>{report.kind === 'intervention' ? (i === 0 ? '介入先' : 'Patch取得元') : `実験 ${i + 1}`}</dt>
            <dd>
              {source.title} · {source.dtype} / {source.device}
            </dd>
          </div>
        ))}
        {report.kind === 'intervention' && (
          <>
            <div>
              <dt>方式 / 観測点</dt>
              <dd>
                {request.mode} /{' '}
                {request.mode === 'residual_patch'
                  ? request.stage
                  : `Layer ${request.layer}${request.mode === 'head_ablation' ? ` · Head ${request.head}` : ''}`}
              </dd>
            </div>
            <div>
              <dt>介入先 / 読出位置</dt>
              <dd>
                {result.positions.join(', ')} / {result.readout_position} · Prefill
              </dd>
            </div>
            <div>
              <dt>強さ / 再計算環境</dt>
              <dd>
                {result.strength} / {result.dtype} · {result.device}
              </dd>
            </div>
          </>
        )}
        {report.kind === 'tsne' && (
          <div>
            <dt>全観測点 / Seed / Perplexity</dt>
            <dd>
              {result.point_count} / {result.seed} / {result.perplexity}
            </dd>
          </div>
        )}
      </dl>
      <code className="fingerprint">Model: {report.model_fingerprint}</code>
    </details>
  );
}

function StudyResult({ report }: { report: any }) {
  const result = report.result;
  if (report.kind === 'intervention')
    return (
      <>
        <dl className="resource-list">
          <div>
            <dt>KL（基準 → 介入後）</dt>
            <dd>{number(result.kl_baseline_to_intervened)}</dd>
          </div>
          <div>
            <dt>Total variation</dt>
            <dd>{number(result.total_variation)}</dd>
          </div>
          <div>
            <dt>Logits差のL2</dt>
            <dd>{number(result.logit_l2)}</dd>
          </div>
          <div>
            <dt>介入解除後のLogits最大誤差</dt>
            <dd>{number(result.restoration_max_logit_error)}</dd>
          </div>
        </dl>
        <div className="difference-table">
          <table>
            <thead>
              <tr>
                <th>Token / ID</th>
                <th>基準確率</th>
                <th>介入後</th>
                <th>差</th>
              </tr>
            </thead>
            <tbody>
              {result.candidates.map((c: any) => (
                <tr key={c.id}>
                  <td>
                    {c.text || c.raw} / {c.id}
                  </td>
                  <td>{number(c.baseline * 100)}%</td>
                  <td>{number(c.intervened * 100)}%</td>
                  <td>{number(c.difference * 100)} pt</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    );
  if (report.kind === 'statistics')
    return (
      <>
        <p className="subtle-note">{result.unit}</p>
        <div className="difference-table">
          <table>
            <thead>
              <tr>
                <th>観測点</th>
                <th>高次元指標</th>
                <th>実験数</th>
                <th>平均</th>
                <th>実験間標準偏差</th>
                <th>最小 / 最大</th>
              </tr>
            </thead>
            <tbody>
              {result.rows.map((r: any) => (
                <tr key={`${r.stage}:${r.metric}`}>
                  <td>{r.stage}</td>
                  <td>{r.metric}</td>
                  <td>{r.experiments}</td>
                  <td>{number(r.mean)}</td>
                  <td>{number(r.std)}</td>
                  <td>
                    {number(r.min)} / {number(r.max)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    );
  return <NonlinearScene result={result} />;
}

function NonlinearScene({ result }: { result: any }) {
  const [stage, setStage] = useState(0);
  const [selected, setSelected] = useState<number[]>([0]);
  const [lowPower, setLowPower] = useState(false);
  const [data, setData] = useState(() => new Float32Array(result.coordinates.flat(2)));
  useEffect(() => {
    setData(new Float32Array(result.coordinates.flat(2)));
    setStage(0);
    setSelected([0]);
  }, [result]);
  return (
    <div className="nonlinear-view">
      <p className="warning">
        t-SNEの探索表示 · 非線形座標です。線は位置対応の案内で、実距離・変位を表しません。
      </p>
      <TrajectoryScene
        data={data}
        count={result.tokens.length}
        stages={result.stages.length}
        stage={stage}
        tokens={result.tokens}
        selected={selected}
        onSelect={(i) => setSelected([i])}
        onStage={setStage}
        growth={false}
        intensity={false}
        arrows={false}
        lowPower={lowPower}
        measuredLabel="t-SNE · EXPLORATORY"
      />
      <label className="field">
        t-SNE上の観測点
        <Select aria-label="t-SNE上の観測点" value={stage} onChange={(e) => setStage(Number(e.target.value))}>
          {result.stages.map((s: string, i: number) => (
            <option key={s} value={i}>
              {s}
            </option>
          ))}
        </Select>
      </label>
      <label className="field">
        非線形表示のトークン
        <Select
          aria-label="非線形表示のトークン"
          value={selected[0] ?? 0}
          onChange={(e) => setSelected([Number(e.target.value)])}
        >
          {result.tokens.map((t: any) => (
            <option key={t.position} value={t.position}>
              #{t.position} {t.raw} / ID {t.id}
            </option>
          ))}
        </Select>
      </label>
      <label className="check-field">
        <input type="checkbox" checked={lowPower} onChange={(e) => setLowPower(e.target.checked)} />
        軽量表示（全点維持）
      </label>
      <p>
        全{result.point_count}点 · Seed {result.seed} · Perplexity {result.perplexity} · KL{' '}
        {number(result.kl_divergence)}
      </p>
    </div>
  );
}
