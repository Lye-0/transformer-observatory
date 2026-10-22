import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowDownToLine,
  Check,
  ChevronLeft,
  ChevronRight,
  Focus,
  GitCompareArrows,
  Maximize2,
  Minimize2,
  Pause,
  Pin,
  Play,
  Search,
  ShieldCheck,
  X,
} from 'lucide-react';
import { api, coordinates, number, type Manifest } from './api';
import Select from './Select';
import TrajectoryScene, { tokenColor } from './TrajectoryScene';
import FloatingPanel from './FloatingPanel';
import Heatmap, { HeatmapDetail } from './Heatmap';
import { colorLimit, type TensorValue } from './heatmapMath';

type PinnedValue = {
  id: string;
  context: string;
  space: string;
  tensor?: TensorValue;
  values?: (number | null)[];
};
type ExpandedValue = { id: string; context: string; tensor: TensorValue; pinId?: string };

type Props = {
  id: string;
  onError: (message: string) => void;
  onReanalyze: (request: Record<string, any>, fingerprint: string) => void;
};
const metricLabels = [
  'ベクトルノルム',
  '初期からの変位',
  '直前の更新量',
  '相対変化率',
  '前段階とのcos類似度',
  '3Dでの更新距離',
  'Attention／MLP更新cos',
];
const stageLabel = (value: string) =>
  value === 'embedding'
    ? 'Embedding'
    : value.replace('layer.', 'Block ').replace('.attention', ' / Attention後').replace('.mlp', ' / MLP後');

export default function Explorer({ id, onError, onReanalyze }: Props) {
  const [m, setM] = useState<Manifest | null>(null);
  const [data, setData] = useState<Float32Array | null>(null);
  const [metrics, setMetrics] = useState<Float32Array | undefined>();
  const [view, setView] = useState('displacement');
  const [selected, setSelected] = useState<number[]>([0]);
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [growth, setGrowth] = useState(false);
  const [intensity, setIntensity] = useState(false);
  const [arrows, setArrows] = useState(false);
  const [lowPower, setLowPower] = useState(false);
  const [full, setFull] = useState(false);
  const [search, setSearch] = useState('');
  const [tab, setTab] = useState('Overview');
  const [tokenMetrics, setTokenMetrics] = useState<any>(null);
  const [tensorResult, setTensorResult] = useState<{ key: string; tensors: any[]; error?: boolean } | null>(
    null,
  );
  const [pins, setPins] = useState<PinnedValue[]>([]);
  const [pinsOpen, setPinsOpen] = useState(false);
  const [expanded, setExpanded] = useState<ExpandedValue | null>(null);
  const visualization = useRef<HTMLElement>(null);
  const pinButton = useRef<HTMLButtonElement>(null);
  const [traces, setTraces] = useState<any[]>([]);
  const [compareId, setCompareId] = useState('');
  const [comparison, setComparison] = useState<any>(null);
  const [rightMetrics, setRightMetrics] = useState<Float32Array>();
  const [compareMode, setCompareMode] = useState('overlay');
  const [comparing, setComparing] = useState(false);
  const [verified, setVerified] = useState<boolean | null>(null);
  const cameraBus = useRef<{ position: number[]; target: number[]; source?: string } | null>(null);

  useEffect(() => {
    let active = true;
    setM(null);
    setComparison(null);
    setData(null);
    setPins([]);
    setPinsOpen(false);
    setExpanded(null);
    setSelected([0]);
    setVerified(null);
    setMetrics(undefined);
    setTokenMetrics(null);
    api<Manifest>(`traces/${id}`)
      .then((result) => {
        if (active) {
          setM(result);
          setStage(result.stages.length - 1);
        }
      })
      .catch((e) => onError(String(e)));
    api<any[]>('traces')
      .then(setTraces)
      .catch((e) => onError(String(e)));
    fetch(`/api/traces/${id}/metric-values`)
      .then((r) => (r.ok ? r.arrayBuffer() : undefined))
      .then((a) => {
        if (active && a) setMetrics(new Float32Array(a));
      });
    return () => {
      active = false;
    };
  }, [id]);
  useEffect(() => {
    let active = true;
    coordinates(id, view)
      .then((a) => {
        if (active) setData(a);
      })
      .catch((e) => onError(String(e)));
    return () => {
      active = false;
    };
  }, [id, view]);
  useEffect(() => {
    let active = true;
    api(`traces/${id}/metrics?token=${selected[0] ?? 0}`)
      .then((result) => {
        if (active) setTokenMetrics({ trace: id, token: selected[0] ?? 0, data: result });
      })
      .catch((e) => onError(String(e)));
    return () => {
      active = false;
    };
  }, [id, selected[0]]);
  const token = m?.tokens[selected[0] ?? 0];
  const layer = stage === 0 ? 0 : Math.floor((stage - 1) / 2);
  const substage = m?.stages[stage] || 'embedding';
  const tensorQuery =
    tab === 'Overview' || tab === 'Tensor' ? `kind=residual&stage=${substage}` : `kind=tensor&layer=${layer}`;
  const tensorKey = `${id}:${token?.position}:${token?.step}:${tensorQuery}`;
  const tensors = tensorResult?.key === tensorKey ? tensorResult.tensors : [];
  useEffect(() => {
    if (!m || !token) return;
    let active = true;
    setTensorResult(null);
    api(`traces/${id}/tensor?${tensorQuery}&token=${token.position}&step=${token.step}`)
      .then((result) => {
        if (active) setTensorResult({ key: tensorKey, tensors: result.tensors });
      })
      .catch((e) => {
        if (active) {
          setTensorResult({ key: tensorKey, tensors: [], error: true });
          onError(String(e));
        }
      });
    return () => {
      active = false;
    };
  }, [id, substage, layer, token?.position, tab, m]);
  useEffect(() => {
    if (!playing || !m) return;
    const timer = window.setInterval(
      () => setStage((s) => (s >= m.stages.length - 1 ? 0 : s + 1)),
      750 / speed,
    );
    return () => clearInterval(timer);
  }, [playing, speed, m]);
  const comparisonData = useMemo(() => {
    if (!comparison) return null;
    const key = view === 'scientific' ? 'coordinates' : 'displacement';
    return {
      left: new Float32Array(comparison[key].left.flat(2)),
      right: new Float32Array(comparison[key].right.flat(2)),
    };
  }, [comparison, view]);
  const stageDifferences = useMemo(
    () =>
      new Map<string, any>(
        comparison?.differences
          .filter((d: any) => d.stage === substage)
          .map((d: any) => [`${d.left}:${d.right}`, d]) || [],
      ),
    [comparison, substage],
  );
  const currentMetrics =
    tokenMetrics?.trace === id && tokenMetrics.token === (selected[0] ?? 0)
      ? tokenMetrics.data.values[stage]
      : undefined;
  const pinScale = colorLimit(pins.flatMap((p) => (p.tensor ? [p.tensor.values] : [])));
  const tensorPinCount = pins.filter((p) => p.tensor).length;
  const context = `${token?.text || token?.raw || ''} · #${token?.position ?? 0} · ${stageLabel(substage)} · ${token?.step ? `Decode ${token.step}` : 'Prefill'}`;
  const measurementId = `${id}:${token?.position}:${token?.step}:${substage}`;
  const metricPinId = `${measurementId}:metrics`;
  function addPin(value: PinnedValue) {
    setPins((old) => (old.some((item) => item.id === value.id) ? old : [...old, value]));
  }
  function expandTensor(tensor: TensorValue, label: string, valueId: string, pinId?: string) {
    setExpanded({
      id: valueId,
      context: label,
      tensor: { ...tensor, shape: [...tensor.shape], values: [...tensor.values] },
      pinId,
    });
  }
  function select(index: number, multi = false) {
    setSelected((old) =>
      multi ? (old.includes(index) ? old.filter((i) => i !== index) : [...old, index]) : [index],
    );
  }
  async function compare() {
    setComparing(true);
    try {
      setComparison(await api('traces/compare', 'POST', { left: id, right: compareId }));
      const response = await fetch(`/api/traces/${compareId}/metric-values`);
      setRightMetrics(response.ok ? new Float32Array(await response.arrayBuffer()) : undefined);
    } catch (e) {
      onError(String(e));
    } finally {
      setComparing(false);
    }
  }
  if (!m || !data) return <div className="empty-state">保存された観測データを読み込み中…</div>;
  const n = m.tokens.filter((t) => t.processed).length;
  const visibleTensors = tensors.filter((t) =>
    tab === 'Attention'
      ? /^(q|k|v|attention|head)/.test(t.name)
      : tab === 'MLP'
        ? /^(gate|up|product|activation|mlp)/.test(t.name)
        : true,
  );
  const sceneProps = {
    count: n,
    stages: m.stages.length,
    stage,
    tokens: m.tokens,
    selected,
    growth,
    intensity,
    metrics,
    arrows,
    lowPower,
    onSelect: select,
    onStage: (s: number) => {
      setStage(s);
      setPlaying(false);
    },
  };
  const displayedProjection = comparison?.projection || m.projection;
  return (
    <div className={`explorer ${full ? 'focus-mode' : ''}`}>
      <div className="explorer-toolbar">
        <div className="experiment-title">
          <span className="eyebrow">EXPERIMENT</span>
          <h2 title={m.title}>{m.title}</h2>
          <span className="muted mono">
            {m.model.name} · {m.device.toUpperCase()} · Level {m.level}
          </span>
        </div>
        <div className="explorer-tools">
          <div className="segmented" aria-label="表示モード">
            <button className={view === 'scientific' ? 'active' : ''} onClick={() => setView('scientific')}>
              Scientific
            </button>
            <button
              className={view === 'displacement' ? 'active' : ''}
              onClick={() => setView('displacement')}
            >
              Displacement
            </button>
          </div>
          <button
            className={`button secondary pin-toggle ${pinsOpen ? 'active' : ''}`}
            ref={pinButton}
            aria-haspopup="dialog"
            aria-expanded={pinsOpen}
            aria-controls="pinned-values-panel"
            onClick={() => setPinsOpen(!pinsOpen)}
          >
            <Pin size={15} />
            ピン <span aria-live="polite">{pins.length}</span>
          </button>
          <button
            className="icon-button"
            title={full ? 'パネルを表示' : '3Dを広く表示'}
            onClick={() => setFull(!full)}
          >
            {full ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
        </div>
      </div>
      <div className="explorer-body">
        <aside className="tokens-panel">
          <div className="panel-label">
            TOKENS <span>{n}</span>
          </div>
          <div className="token-search">
            <Search size={15} />
            <input
              aria-label="トークンを検索"
              placeholder="表記・ID・位置で検索"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div className="tokens-list">
            {m.tokens
              .filter(
                (t) =>
                  !search ||
                  `${t.raw} ${t.text} ${t.id} #${t.position}`.toLowerCase().includes(search.toLowerCase()),
              )
              .map((t) => (
                <button
                  className={selected.includes(t.position) ? 'token-row active' : 'token-row'}
                  key={t.position}
                  disabled={!t.processed}
                  onClick={(e) => select(t.position, e.ctrlKey || e.metaKey)}
                  aria-pressed={selected.includes(t.position)}
                >
                  <i style={{ background: tokenColor(t.position) }} />
                  <small>{String(t.position).padStart(3, '0')}</small>
                  <span>{t.raw}</span>
                  <em>{t.generated ? 'GEN' : t.id}</em>
                </button>
              ))}
          </div>
          <div className="panel-footnote">
            Ctrl＋クリックで複数選択
            <br />
            選択していない軌跡も維持します。
          </div>
        </aside>
        <section className="visualization" ref={visualization}>
          <div className="view-explanation">
            <b>{view === 'displacement' ? 'Δx = x(s) − x(0)' : 'P · (x − μ)'}</b>
            <span>
              {view === 'displacement'
                ? '各トークンの初期ベクトルを原点とした、実際の変位'
                : '全トークン・全観測点を、共通の固定基底へ射影'}
            </span>
          </div>
          <div className={comparison && compareMode === 'side' ? 'split-scenes' : 'scenes'}>
            <TrajectoryScene
              {...sceneProps}
              data={comparisonData?.left || data}
              overlay={comparisonData && compareMode === 'overlay' ? comparisonData.right : undefined}
              cameraBus={comparison && compareMode === 'side' ? cameraBus : undefined}
            />
            {comparisonData && compareMode === 'side' && (
              <TrajectoryScene
                {...sceneProps}
                count={comparison.right_tokens.filter((t: any) => t.processed).length}
                tokens={comparison.right_tokens}
                data={comparisonData.right}
                metrics={rightMetrics}
                selected={[]}
                onSelect={() => {}}
                cameraBus={cameraBus}
              />
            )}
          </div>
          <div className="scene-options">
            <label>
              <input type="checkbox" checked={growth} onChange={(e) => setGrowth(e.target.checked)} />
              段階まで伸ばす
            </label>
            <label>
              <input type="checkbox" checked={intensity} onChange={(e) => setIntensity(e.target.checked)} />
              更新量で明るさ
            </label>
            <label>
              <input type="checkbox" checked={arrows} onChange={(e) => setArrows(e.target.checked)} />
              選択位置の更新矢印
            </label>
            <label>
              <input type="checkbox" checked={lowPower} onChange={(e) => setLowPower(e.target.checked)} />
              軽量表示
            </label>
          </div>
          <div className="projection-note">
            PCA寄与率{' '}
            {(
              displayedProjection.explained_variance_ratio.reduce((a: number, b: number) => a + b, 0) * 100
            ).toFixed(1)}
            % · {displayedProjection.algorithm} · 3D距離と高次元距離は異なります
            {intensity && ' · 明るさ：高次元更新量の対数尺度（実験内基準）'}
          </div>
        </section>
        <aside className="inspector">
          <div className="panel-label">
            INSPECTOR <span>#{token?.position ?? '—'}</span>
          </div>
          <div className="inspector-token">
            <span style={{ color: tokenColor(token?.position || 0) }}>{token?.raw}</span>
            <small>
              ID {token?.id} · {m.model.hidden_size} dimensions ·{' '}
              {token?.step ? `Decode ${token.step}` : 'Prefill'}
            </small>
          </div>
          <div className="inspector-tabs">
            {['Overview', 'Attention', 'MLP', 'Tensor', 'Prediction'].map((t) => (
              <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>
                {t}
              </button>
            ))}
          </div>
          <div className="inspector-content">
            <p className="mono stage-caption">{stageLabel(substage)}</p>
            {tab === 'Overview' && (
              <>
                <div className="metric-cards">
                  {metricLabels.map((label, i) => (
                    <div key={label}>
                      <span>{comparison && i === 5 ? '元実験3Dでの更新距離' : label}</span>
                      <b>{number(currentMetrics?.[i])}</b>
                    </div>
                  ))}
                </div>
                <p className="subtle-note">
                  指標は元の高次元値から計算しています。更新量の大きさだけで因果的重要性は判断できません。
                </p>
                <button
                  className="secondary compact"
                  disabled={!currentMetrics || pins.some((p) => p.id === metricPinId)}
                  onClick={() =>
                    addPin({
                      id: metricPinId,
                      context,
                      values: metricLabels.map((_, i) => currentMetrics?.[i] ?? null),
                      space: 'residual',
                    })
                  }
                >
                  <Pin size={14} />
                  {pins.some((p) => p.id === metricPinId) ? '指標はピン済み' : '指標をPin'}
                </button>
              </>
            )}
            {tab === 'Attention' && (
              <div className="subtle-note">
                Query {m.model.heads} heads → KV {m.model.kv_heads}{' '}
                heads。Q/KはRoPE適用段階を区別します。Attention係数は因果的重要度ではありません。Head寄与はW_Oを通したResidual空間の値です。
              </div>
            )}
            {tab === 'MLP' && (
              <div className="subtle-note">
                {m.model.architecture === 'gpt2' ? 'Up → GELU → Down' : 'Gate → SiLU × Up → Down'}
                。中間空間とResidual空間を区別します。
              </div>
            )}
            {tab === 'Prediction' ? (
              <Prediction
                id={id}
                stage={substage}
                token={token?.position || 0}
                step={token?.step || 0}
                level={m.level}
                onError={onError}
                fingerprint={m.model.fingerprint}
              />
            ) : (
              <>
                {visibleTensors.length ? (
                  visibleTensors.map((t) => (
                    <details className="tensor-block" key={t.name} open={visibleTensors.length === 1}>
                      <summary>
                        {t.name}
                        <span>{t.shape.join(' × ')}</span>
                      </summary>
                      <span className="space-label">{t.space}</span>
                      <p className="subtle-note">
                        {t.provenance === 'reconstructed'
                          ? '取得した内部値と重みから再構成し、モデル演算との誤差を検証した値です。'
                          : t.provenance === 'backend_mask'
                            ? '適用Mask（1＝遮断、0＝接続可能）'
                            : t.provenance === 'hook'
                              ? '実際のモジュール入出力から取得した値です。'
                              : 'この保存形式には取得経路の区別がありません。'}
                        {t.heads && ` · Head ${t.heads.join(', ')}`}
                      </p>
                      <div className="tensor-stats">
                        <span>min {number(t.min)}</span>
                        <span>max {number(t.max)}</span>
                        <span>norm {number(t.norm)}</span>
                        <span>mean {number(t.mean)}</span>
                        <span>std {number(t.std)}</span>
                      </div>
                      <Heatmap
                        key={`${measurementId}:${t.name}`}
                        tensor={t}
                        context={`${context} · ${t.name}`}
                        onExpand={() =>
                          expandTensor(t, `${context} · ${t.name}`, `${measurementId}:${t.name}`)
                        }
                      />
                      {t.name === 'attention_weights' && <HeadConnections tensor={t} tokens={m.tokens} />}
                      <p className="subtle-note">
                        {t.dimensions} 成分
                        {t.truncated && ' · ヒートマップは先頭16,384成分。軌跡は省略していません。'}
                      </p>
                      <button
                        className="text-button"
                        disabled={pins.some((p) => p.id === `${measurementId}:${t.name}`)}
                        onClick={() =>
                          addPin({
                            id: `${measurementId}:${t.name}`,
                            context: `${context} · ${t.name}`,
                            tensor: { ...t, shape: [...t.shape], values: [...t.values] },
                            space: t.space,
                          })
                        }
                      >
                        <Pin size={13} />
                        {pins.some((p) => p.id === `${measurementId}:${t.name}`)
                          ? 'この値はピン済み'
                          : 'この値をPin'}
                      </button>
                      <details>
                        <summary>成分を数値で表示</summary>
                        <pre className="tensor-numbers">
                          {t.values.map((x: number, i: number) => `${i}: ${number(x)}`).join('\n')}
                        </pre>
                      </details>
                    </details>
                  ))
                ) : tensorResult?.key !== tensorKey ? (
                  <p className="subtle-note" role="status">
                    観測値を読み込み中…
                  </p>
                ) : tensorResult.error ? (
                  <p className="warning" role="alert">
                    観測値を読み込めませんでした。
                  </p>
                ) : (
                  <div className="not-captured">
                    <DatabaseIcon />
                    <b>この値は未取得です</b>
                    <p>Level 3／4と対象Layerを選択して追加解析すると、詳細を調べられます。</p>
                    <button
                      className="secondary compact"
                      onClick={() =>
                        onReanalyze(
                          {
                            ...m.request,
                            level: Math.max(3, m.level),
                            layers: [layer],
                            tensors: [],
                            title: `${m.title.slice(0, 130)} · 詳細再解析`,
                          },
                          m.model.fingerprint,
                        )
                      }
                    >
                      このLayerを追加解析
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
        </aside>
      </div>
      {pinsOpen && (
        <FloatingPanel
          id="pinned-values-panel"
          title="ピン比較"
          anchor={visualization}
          fallbackFocus={pinButton}
          onClose={() => setPinsOpen(false)}
        >
          <p className="subtle-note">ピンした時点の観測値です。タブや観測点を切り替えながら確認できます。</p>
          {!pins.length && (
            <div className="pin-empty">
              <Pin size={24} />
              <b>ピンした値はここに表示されます</b>
              <p>Inspectorの「指標をPin」「この値をPin」で追加できます。</p>
            </div>
          )}
          {pins.map((p) => (
            <article className="pinned-value" key={p.id}>
              <header>
                <b>{p.context}</b>
                <button
                  className="icon-button"
                  aria-label={`${p.context}のピンを解除`}
                  title="このピンを解除"
                  onClick={() => setPins((old) => old.filter((item) => item.id !== p.id))}
                >
                  <X size={15} />
                </button>
              </header>
              <span className="space-label">
                {p.space}
                {p.tensor && ` · 形状 [${p.tensor.shape.join(' × ')}]`}
              </span>
              {p.tensor ? (
                <>
                  <Heatmap
                    tensor={p.tensor}
                    context={p.context}
                    scale={tensorPinCount > 1 ? pinScale : undefined}
                    onExpand={() => expandTensor(p.tensor!, p.context, p.id, p.id)}
                  />
                  <p className="subtle-note">
                    {p.tensor.dimensions.toLocaleString()}成分
                    {p.tensor.truncated && ` · 表示は先頭${p.tensor.values.length.toLocaleString()}成分`}
                  </p>
                </>
              ) : (
                <dl className="pinned-metrics">
                  {p.values?.map((value, i) => (
                    <div key={metricLabels[i]}>
                      <dt>{metricLabels[i]}</dt>
                      <dd>{number(value)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </article>
          ))}
          {tensorPinCount > 1 && (
            <p className="subtle-note">
              色範囲はピンしたテンソルの表示値から共通で計算します。同じ空間名でも、層ごとのQ/Kなどの意味的対応は自動的に保証されません。空間・形状も確認してください。
            </p>
          )}
        </FloatingPanel>
      )}
      {expanded && (
        <FloatingPanel
          title="成分を詳しく見る"
          anchor={visualization}
          fallbackFocus={pinButton}
          width={620}
          height={680}
          onClose={() => setExpanded(null)}
        >
          <HeatmapDetail
            key={expanded.id}
            tensor={expanded.tensor}
            context={expanded.context}
            scale={
              expanded.pinId && pins.some((p) => p.id === expanded.pinId) && tensorPinCount > 1
                ? pinScale
                : undefined
            }
          />
        </FloatingPanel>
      )}
      <div className="timeline">
        <button
          className="icon-button"
          title={playing ? '再生を停止' : '観測点を再生'}
          onClick={() => setPlaying(!playing)}
        >
          {playing ? <Pause size={19} /> : <Play size={19} />}
        </button>
        <button
          className="icon-button"
          title="前の観測点"
          onClick={() => setStage((s) => Math.max(0, s - 1))}
        >
          <ChevronLeft size={17} />
        </button>
        <input
          aria-label="Layer・内部段階"
          type="range"
          min={0}
          max={m.stages.length - 1}
          value={stage}
          onChange={(e) => {
            setPlaying(false);
            setStage(Number(e.target.value));
          }}
        />
        <button
          className="icon-button"
          title="次の観測点"
          onClick={() => setStage((s) => Math.min(m.stages.length - 1, s + 1))}
        >
          <ChevronRight size={17} />
        </button>
        <span className="timeline-stage">
          {stageLabel(substage)}
          <small>
            {stage + 1} / {m.stages.length} 観測点
          </small>
        </span>
        <Select
          aria-label="生成ステップ"
          value={token?.step || 0}
          onChange={(e) => {
            const t = m.tokens.find((t) => t.step === Number(e.target.value));
            if (t) select(t.position);
          }}
        >
          <option value={0}>Prefill</option>
          {m.tokens
            .filter((t) => t.generated)
            .map((t) => (
              <option key={t.position} value={t.step}>
                Decode {t.step}
              </option>
            ))}
        </Select>
        <Select aria-label="再生速度" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          <option value={0.5}>0.5×</option>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
          <option value={4}>4×</option>
        </Select>
      </div>
      <div className="explorer-footer">
        <span>
          <ShieldCheck size={14} />
          {m.integrity?.status === 'verified' ? '数値チェック合格' : '一部検証'} · {m.integrity?.checks}{' '}
          checks
        </span>
        <button
          className="text-button"
          onClick={async () => {
            try {
              setVerified((await api(`traces/${id}/verify`)).verified);
            } catch (e) {
              onError(String(e));
            }
          }}
        >
          {verified === true ? <Check size={14} /> : <Focus size={14} />}
          {verified === null ? '保存データを検証' : verified ? 'ハッシュ一致' : '不一致を検出'}
        </button>
        <a href={`/api/traces/${id}/export`}>
          <ArrowDownToLine size={14} />
          エクスポート
        </a>
        <span>観測点間の再生は表示上の操作です。</span>
      </div>
      <details className="comparison-panel">
        <summary>
          <GitCompareArrows size={16} />
          プロンプト比較<span>同じモデルを、共通の座標空間で。</span>
        </summary>
        <div className="comparison-body">
          <div className="button-row">
            <Select
              aria-label="比較する実験"
              value={compareId}
              onChange={(e) => setCompareId(e.target.value)}
            >
              <option value="">比較する実験を選択</option>
              {traces
                .filter((t) => t.id !== id && t.status === 'completed')
                .map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.title}
                  </option>
                ))}
            </Select>
            <button className="secondary" disabled={!compareId || comparing} onClick={compare}>
              {comparing ? '共通PCAを計算中…' : '比較する'}
            </button>
            {comparison && (
              <>
                <Select
                  aria-label="比較表示"
                  value={compareMode}
                  onChange={(e) => setCompareMode(e.target.value)}
                >
                  <option value="overlay">Overlay</option>
                  <option value="side">Side by Side</option>
                  <option value="difference">Difference Analysis</option>
                </Select>
                <button className="text-button" onClick={() => setComparison(null)}>
                  比較を閉じる
                </button>
              </>
            )}
          </div>
          {comparison && (
            <>
              <p className="subtle-note">
                両実験のResidualから共通PCAを計算しました。左＝トークン色、右＝金色（Overlay）。共通接頭辞{' '}
                {comparison.common_prefix_tokens} tokens · 高次元差の最大{' '}
                {number(comparison.common_prefix_max_distance)}
              </p>
              {comparison.settings_differ && (
                <p className="warning">
                  デバイス・精度・取得方式の設定に違いがあります。差分には計算条件の影響が含まれます。
                </p>
              )}
              <p className="subtle-note">
                左 {comparison.left_tokens.length} tokens ／ 右 {comparison.right_tokens.length}{' '}
                tokens。挿入・削除は差分値を計算せず、対応表に残します。Side by
                Sideの右側はトークン色で表示します。
              </p>
              <button
                className="text-button"
                onClick={() => {
                  const url = URL.createObjectURL(
                    new Blob([JSON.stringify({ schema_version: 1, ...comparison }, null, 2)], {
                      type: 'application/json',
                    }),
                  );
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `comparison-${id}-${compareId}.json`;
                  a.click();
                  setTimeout(() => URL.revokeObjectURL(url), 1000);
                }}
              >
                共通基底・対応表・差分をJSONで保存
              </button>
              <div className="difference-table">
                <table>
                  <thead>
                    <tr>
                      <th>左位置／ID</th>
                      <th>右位置／ID</th>
                      <th>対応</th>
                      <th>高次元距離</th>
                      <th>cos類似度</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.alignment.map((pair: any) => {
                      const d = stageDifferences.get(`${pair.left}:${pair.right}`);
                      return (
                        <tr key={`${pair.left}:${pair.right}`}>
                          <td>
                            {pair.left === null
                              ? '—'
                              : `#${pair.left} / ${comparison.left_tokens[pair.left].id}`}
                          </td>
                          <td>
                            {pair.right === null
                              ? '—'
                              : `#${pair.right} / ${comparison.right_tokens[pair.right].id}`}
                          </td>
                          <td>
                            {pair.match
                              ? '同一Token ID'
                              : pair.left === null
                                ? '挿入'
                                : pair.right === null
                                  ? '削除'
                                  : '置換位置'}
                          </td>
                          <td>{number(d?.distance)}</td>
                          <td>{number(d?.cosine)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      </details>
    </div>
  );
}

function HeadConnections({ tensor, tokens }: { tensor: any; tokens: Manifest['tokens'] }) {
  const [head, setHead] = useState(0);
  const width = tensor.shape[1];
  const safeHead = Math.min(head, tensor.shape[0] - 1);
  const values: { position: number; value: number }[] = tensor.values
    .slice(safeHead * width, (safeHead + 1) * width)
    .map((value: number, position: number) => ({ position, value }));
  const sorted = values.sort((a, b) => b.value - a.value).slice(0, 16);
  return (
    <div className="attention-connections">
      <label className="field">
        接続を調べるQuery Head
        <Select
          aria-label="接続を調べるQuery Head"
          value={safeHead}
          onChange={(e) => setHead(Number(e.target.value))}
        >
          {Array.from({ length: tensor.shape[0] }, (_, i) => (
            <option key={i} value={i}>
              Head {tensor.heads?.[i] ?? i}
            </option>
          ))}
        </Select>
      </label>
      <p className="subtle-note">
        選択Query位置から各Key位置への係数。上位16接続を表に表示します。行列の全値は成分表示で確認できます。
      </p>
      {sorted.map(({ position, value }) => (
        <div className="prediction-row" key={position}>
          <span>
            #{position} {tokens[position]?.raw}
          </span>
          <b>{number(value)}</b>
        </div>
      ))}
      {values.length < width && (
        <p className="warning">
          このHeadの全成分は表示上限を超えています。完全な値はエクスポートで確認してください。
        </p>
      )}
    </div>
  );
}

function DatabaseIcon() {
  return <span className="mono">∅</span>;
}

function Prediction({
  id,
  stage,
  token,
  step,
  level,
  onError,
  fingerprint,
}: {
  id: string;
  stage: string;
  token: number;
  step: number;
  level: number;
  onError: (s: string) => void;
  fingerprint: string;
}) {
  const [result, setResult] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [lenses, setLenses] = useState<any[]>([]);
  const [lensId, setLensId] = useState('');
  const current = useRef('');
  current.current = `${id}:${stage}:${token}:${lensId}`;
  useEffect(() => {
    let active = true;
    api<any[]>('research/lenses')
      .then((items) => {
        if (active) setLenses(items.filter((l) => l.model_fingerprint === fingerprint));
      })
      .catch((e) => onError(String(e)));
    return () => {
      active = false;
    };
  }, [fingerprint]);
  useEffect(() => {
    setResult(null);
  }, [id, stage, token, lensId]);
  return (
    <>
      <p className="subtle-note">
        Logit Lensは、保存した途中層のResidualに最終NormとLM
        Headを適用した読み出しです。途中層でモデルが実際に出した確率とは異なります。
      </p>
      <label className="field">
        読み出し方式
        <Select aria-label="読み出し方式" value={lensId} onChange={(e) => setLensId(e.target.value)}>
          <option value="">Logit Lens</option>
          {lenses.map((l) => (
            <option key={l.id} value={l.id}>
              Tuned Lens · {l.name}
            </option>
          ))}
        </Select>
      </label>
      {lensId && (
        <p className="subtle-note">
          学習済みの層別残差アフィン変換を適用します。学習範囲外での精度は保証しません。EmbeddingとBlock出力に対応します。
        </p>
      )}
      {!lenses.length && (
        <p className="subtle-note">Tuned LensはResearch画面で学習済み重みを登録すると利用できます。</p>
      )}
      <button
        className="secondary compact"
        disabled={level < 2 || busy || Boolean(lensId && stage.endsWith('.attention'))}
        onClick={async () => {
          setBusy(true);
          const key = current.current;
          try {
            const response = await api(`traces/${id}/lens`, 'POST', {
              stage,
              token,
              step,
              lens_id: lensId || null,
            });
            if (current.current === key) setResult(response);
          } catch (e) {
            onError(String(e));
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? 'モデルを読み込んで計算中…' : lensId ? 'Tuned Lensを計算' : 'Logit Lensを計算'}
      </button>
      {level < 2 && <p className="subtle-note">Level 2以上のResidualが必要です。</p>}
      {result?.candidates.map((c: any) => (
        <div className="prediction-row" key={c.id}>
          <span>{c.text || c.raw}</span>
          <small>ID {c.id}</small>
          <b>{(c.probability * 100).toFixed(2)}%</b>
        </div>
      ))}
    </>
  );
}
