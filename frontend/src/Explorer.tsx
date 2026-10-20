import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDownToLine, Check, ChevronLeft, ChevronRight, Focus, GitCompareArrows, Maximize2, Minimize2, Pause, Pin, Play, Search, ShieldCheck, X } from 'lucide-react';
import { api, coordinates, number, type Manifest } from './api';
import TrajectoryScene, { tokenColor } from './TrajectoryScene';

type Props = { id: string; onError: (message: string) => void };
const metricLabels = ['ベクトルノルム', '初期からの変位', '直前の更新量', '相対変化率', '前段階とのcos類似度', '3Dでの更新距離', 'Attention／MLP更新cos'];
const stageLabel = (value: string) => value === 'embedding' ? 'Embedding' : value.replace('layer.', 'Block ').replace('.attention', ' / Attention後').replace('.mlp', ' / MLP後');

export default function Explorer({ id, onError }: Props) {
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
  const [tensors, setTensors] = useState<any[]>([]);
  const [pins, setPins] = useState<any[]>([]);
  const [traces, setTraces] = useState<any[]>([]);
  const [compareId, setCompareId] = useState('');
  const [comparison, setComparison] = useState<any>(null);
  const [compareMode, setCompareMode] = useState('overlay');
  const [comparing, setComparing] = useState(false);
  const [verified, setVerified] = useState<boolean | null>(null);
  const cameraBus = useRef<{ position: number[]; target: number[] } | null>(null);

  useEffect(() => {
    let active = true;
    setM(null); setComparison(null); setData(null); setPins([]); setSelected([0]); setVerified(null);
    api<Manifest>(`traces/${id}`).then(result => { if (active) { setM(result); setStage(result.stages.length - 1); } }).catch(e => onError(String(e)));
    api<any[]>('traces').then(setTraces).catch(e => onError(String(e)));
    fetch(`/api/traces/${id}/metric-values`).then(r => r.ok ? r.arrayBuffer() : undefined).then(a => { if (active && a) setMetrics(new Float32Array(a)); });
    return () => { active = false; };
  }, [id]);
  useEffect(() => { let active = true; coordinates(id, view).then(a => { if (active) setData(a); }).catch(e => onError(String(e))); return () => { active = false; }; }, [id, view]);
  useEffect(() => {
    let active = true;
    api(`traces/${id}/metrics?token=${selected[0] ?? 0}`).then(result => { if (active) setTokenMetrics(result); }).catch(e => onError(String(e)));
    return () => { active = false; };
  }, [id, selected[0]]);
  const token = m?.tokens[selected[0] ?? 0];
  const layer = stage === 0 ? 0 : Math.floor((stage - 1) / 2);
  const substage = m?.stages[stage] || 'embedding';
  useEffect(() => {
    if (!m || !token) return;
    let active = true; setTensors([]);
    const q = tab === 'Overview' || tab === 'Tensor' ? `kind=residual&stage=${substage}` : `kind=tensor&layer=${layer}`;
    api(`traces/${id}/tensor?${q}&token=${token.position}&step=${token.step}`).then(result => { if (active) setTensors(result.tensors); }).catch(e => onError(String(e)));
    return () => { active = false; };
  }, [id, substage, layer, token?.position, tab, m]);
  useEffect(() => {
    if (!playing || !m) return;
    const timer = window.setInterval(() => setStage(s => s >= m.stages.length - 1 ? 0 : s + 1), 750 / speed);
    return () => clearInterval(timer);
  }, [playing, speed, m]);
  const comparisonData = useMemo(() => {
    if (!comparison) return null;
    const key = view === 'scientific' ? 'coordinates' : 'displacement';
    return { left: new Float32Array(comparison[key].left.flat(2)), right: new Float32Array(comparison[key].right.flat(2)) };
  }, [comparison, view]);
  const currentMetrics = tokenMetrics?.values[stage];
  function select(index: number, multi = false) { setSelected(old => multi ? old.includes(index) ? old.filter(i => i !== index) : [...old, index] : [index]); }
  async function compare() {
    setComparing(true);
    try { setComparison(await api('traces/compare', 'POST', { left: id, right: compareId })); }
    catch (e) { onError(String(e)); } finally { setComparing(false); }
  }
  if (!m || !data) return <div className="empty-state">保存された観測データを読み込み中…</div>;
  const n = m.tokens.filter(t => t.processed).length;
  const visibleTensors = tensors.filter(t => tab === 'Attention' ? /^(q|k|v|attention|head)/.test(t.name) : tab === 'MLP' ? /^(gate|up|product|activation|mlp)/.test(t.name) : true);
  const sceneProps = { count: n, stages: m.stages.length, stage, tokens: m.tokens, selected, growth, intensity, metrics, arrows, lowPower, onSelect: select };
  return <div className={`explorer ${full ? 'focus-mode' : ''}`}>
    <div className="explorer-toolbar"><div className="experiment-title"><span className="eyebrow">EXPERIMENT</span><h2 title={m.title}>{m.title}</h2><span className="muted mono">{m.model.name} · {m.device.toUpperCase()} · Level {m.level}</span></div><div className="explorer-tools"><div className="segmented" aria-label="表示モード"><button className={view === 'scientific' ? 'active' : ''} onClick={() => setView('scientific')}>Scientific</button><button className={view === 'displacement' ? 'active' : ''} onClick={() => setView('displacement')}>Displacement</button></div><button className="icon-button" title={full ? 'パネルを表示' : '3Dを広く表示'} onClick={() => setFull(!full)}>{full ? <Minimize2 size={18} /> : <Maximize2 size={18} />}</button></div></div>
    <div className="explorer-body"><aside className="tokens-panel"><div className="panel-label">TOKENS <span>{n}</span></div><div className="token-search"><Search size={15} /><input aria-label="トークンを検索" placeholder="表記・ID・位置で検索" value={search} onChange={e => setSearch(e.target.value)} /></div><div className="tokens-list">{m.tokens.filter(t => !search || `${t.raw} ${t.text} ${t.id} #${t.position}`.toLowerCase().includes(search.toLowerCase())).map(t => <button className={selected.includes(t.position) ? 'token-row active' : 'token-row'} key={t.position} disabled={!t.processed} onClick={e => select(t.position, e.ctrlKey || e.metaKey)} aria-pressed={selected.includes(t.position)}><i style={{ background: tokenColor(t.position) }} /><small>{String(t.position).padStart(3, '0')}</small><span>{t.raw}</span><em>{t.generated ? 'GEN' : t.id}</em></button>)}</div><div className="panel-footnote">Ctrl＋クリックで複数選択<br />選択していない軌跡も維持します。</div></aside>
    <section className="visualization"><div className="view-explanation"><b>{view === 'displacement' ? 'Δx = x(s) − x(0)' : 'P · (x − μ)'}</b><span>{view === 'displacement' ? '各トークンの初期ベクトルを原点とした、実際の変位' : '全トークン・全観測点を、共通の固定基底へ射影'}</span></div>
      <div className={comparison && compareMode === 'side' ? 'split-scenes' : 'scenes'}><TrajectoryScene {...sceneProps} data={comparisonData?.left || data} overlay={comparisonData && compareMode === 'overlay' ? comparisonData.right : undefined} cameraBus={comparison && compareMode === 'side' ? cameraBus : undefined} />{comparisonData && compareMode === 'side' && <TrajectoryScene {...sceneProps} count={comparison.right_tokens.filter((t: any) => t.processed).length} tokens={comparison.right_tokens} data={comparisonData.right} selected={[]} onSelect={() => {}} cameraBus={cameraBus} />}</div>
      <div className="scene-options"><label><input type="checkbox" checked={growth} onChange={e => setGrowth(e.target.checked)} />段階まで伸ばす</label><label><input type="checkbox" checked={intensity} onChange={e => setIntensity(e.target.checked)} />更新量で明るさ</label><label><input type="checkbox" checked={arrows} onChange={e => setArrows(e.target.checked)} />選択位置の更新矢印</label><label><input type="checkbox" checked={lowPower} onChange={e => setLowPower(e.target.checked)} />軽量表示</label></div>
      <div className="projection-note">PCA寄与率 {(m.projection.explained_variance_ratio.reduce((a, b) => a + b, 0) * 100).toFixed(1)}% · {m.projection.algorithm} · 3D距離と高次元距離は異なります{intensity && ' · 明るさ：高次元更新量の対数尺度（実験内基準）'}</div>
    </section>
    <aside className="inspector"><div className="panel-label">INSPECTOR <span>#{token?.position ?? '—'}</span></div><div className="inspector-token"><span style={{ color: tokenColor(token?.position || 0) }}>{token?.raw}</span><small>ID {token?.id} · {m.model.hidden_size} dimensions</small></div><div className="inspector-tabs">{['Overview', 'Attention', 'MLP', 'Tensor', 'Prediction'].map(t => <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>)}</div><div className="inspector-content"><p className="mono stage-caption">{stageLabel(substage)}</p>
      {tab === 'Overview' && <><div className="metric-cards">{metricLabels.map((label, i) => <div key={label}><span>{label}</span><b>{number(currentMetrics?.[i])}</b></div>)}</div><p className="subtle-note">指標は元の高次元値から計算しています。更新量の大きさだけで因果的重要性は判断できません。</p><button className="secondary compact" onClick={() => setPins(old => [...old.slice(-1), { label: `${token?.raw} · ${stageLabel(substage)}`, values: [...(currentMetrics || [])], space: 'residual' }])}><Pin size={14} />指標をPin</button></>}
      {tab === 'Attention' && <div className="subtle-note">Query {m.model.heads} heads → KV {m.model.kv_heads} heads。Q/KはRoPE適用段階を区別します。Attention係数は因果的重要度ではありません。Head寄与はW_Oを通したResidual空間の値です。</div>}
      {tab === 'MLP' && <div className="subtle-note">{m.model.architecture === 'gpt2' ? 'Up → GELU → Down' : 'Gate → SiLU × Up → Down'}。中間空間とResidual空間を区別します。</div>}
      {tab === 'Prediction' ? <Prediction id={id} stage={substage} token={token?.position || 0} step={token?.step || 0} level={m.level} onError={onError} /> : <>{visibleTensors.length ? visibleTensors.map(t => <details className="tensor-block" key={t.name} open={visibleTensors.length === 1}><summary>{t.name}<span>{t.shape.join(' × ')}</span></summary><span className="space-label">{t.space}</span><div className="tensor-stats"><span>min {number(t.min)}</span><span>max {number(t.max)}</span><span>norm {number(t.norm)}</span></div><Heatmap values={t.values} /><p className="subtle-note">{t.dimensions} 成分{t.truncated && ' · ヒートマップは先頭16,384成分。軌跡は省略していません。'}</p><button className="text-button" onClick={() => setPins(old => [...old.slice(-1), { label: `${token?.raw} · ${substage} · ${t.name}`, tensor: t, space: t.space }])}><Pin size={13} />この値をPin</button><details><summary>成分を数値で表示</summary><pre className="tensor-numbers">{t.values.map((x: number, i: number) => `${i}: ${number(x)}`).join('\n')}</pre></details></details>) : <div className="not-captured"><DatabaseIcon /><b>この値は未取得です</b><p>Level 3／4と対象Layerを選択して追加解析すると、詳細を調べられます。</p></div>}</>}
      {pins.length > 0 && <section className="pins"><div className="panel-label">PIN比較<button className="icon-button" title="Pinを解除" onClick={() => setPins([])}><X size={14} /></button></div>{pins.map((p, i) => <div key={i}><b>{p.label}</b><span className="space-label">{p.space}</span>{p.tensor ? <Heatmap values={p.tensor.values} /> : <small>{p.values.map(number).join(' · ')}</small>}</div>)}{pins.length === 2 && <p className="subtle-note">同じ空間名でも、層ごとのQ/Kなどの意味的対応は自動的に保証されません。</p>}</section>}
    </div></aside></div>
    <div className="timeline"><button className="icon-button" title={playing ? '再生を停止' : '観測点を再生'} onClick={() => setPlaying(!playing)}>{playing ? <Pause size={19} /> : <Play size={19} />}</button><button className="icon-button" title="前の観測点" onClick={() => setStage(s => Math.max(0, s - 1))}><ChevronLeft size={17} /></button><input aria-label="Layer・内部段階" type="range" min={0} max={m.stages.length - 1} value={stage} onChange={e => { setPlaying(false); setStage(Number(e.target.value)); }} /><button className="icon-button" title="次の観測点" onClick={() => setStage(s => Math.min(m.stages.length - 1, s + 1))}><ChevronRight size={17} /></button><span className="timeline-stage">{stageLabel(substage)}<small>{stage + 1} / {m.stages.length} 観測点</small></span><select aria-label="再生速度" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option></select></div>
    <div className="explorer-footer"><span><ShieldCheck size={14} />{m.integrity?.status === 'verified' ? '数値チェック合格' : '一部検証'} · {m.integrity?.checks} checks</span><button className="text-button" onClick={async () => { try { setVerified((await api(`traces/${id}/verify`)).verified); } catch (e) { onError(String(e)); } }}>{verified === true ? <Check size={14} /> : <Focus size={14} />}{verified === null ? '保存データを検証' : verified ? 'ハッシュ一致' : '不一致を検出'}</button><a href={`/api/traces/${id}/export`}><ArrowDownToLine size={14} />エクスポート</a><span>観測点間の再生は表示上の操作です。</span></div>
    <details className="comparison-panel"><summary><GitCompareArrows size={16} />プロンプト比較<span>同じモデルを、共通の座標空間で。</span></summary><div className="comparison-body"><div className="button-row"><select aria-label="比較する実験" value={compareId} onChange={e => setCompareId(e.target.value)}><option value="">比較する実験を選択</option>{traces.filter(t => t.id !== id && t.status === 'completed').map(t => <option key={t.id} value={t.id}>{t.title}</option>)}</select><button className="secondary" disabled={!compareId || comparing} onClick={compare}>{comparing ? '共通PCAを計算中…' : '比較する'}</button>{comparison && <><select aria-label="比較表示" value={compareMode} onChange={e => setCompareMode(e.target.value)}><option value="overlay">Overlay</option><option value="side">Side by Side</option><option value="difference">Difference Analysis</option></select><button className="text-button" onClick={() => setComparison(null)}>比較を閉じる</button></>}</div>{comparison && <><p className="subtle-note">両実験のResidualから共通PCAを計算しました。左＝トークン色、右＝金色（Overlay）。共通接頭辞 {comparison.common_prefix_tokens} tokens · 高次元差の最大 {number(comparison.common_prefix_max_distance)}</p><div className="difference-table"><table><thead><tr><th>左位置／ID</th><th>右位置／ID</th><th>対応</th><th>高次元距離</th><th>cos類似度</th></tr></thead><tbody>{comparison.differences.filter((d: any) => d.stage === substage).map((d: any) => <tr key={`${d.left}:${d.right}`}><td>#{d.left} / {comparison.left_tokens[d.left].id}</td><td>#{d.right} / {comparison.right_tokens[d.right].id}</td><td>{d.match ? '同一Token ID' : '置換位置'}</td><td>{number(d.distance)}</td><td>{number(d.cosine)}</td></tr>)}</tbody></table></div></>}</div></details>
  </div>;
}

function Heatmap({ values }: { values: number[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!, ctx = canvas.getContext('2d'); if (!ctx || !values.length) return;
    const cols = Math.min(64, values.length), rows = Math.ceil(values.length / cols);
    canvas.width = cols * 4; canvas.height = rows * 4;
    const max = Math.max(...values.map(Math.abs), .00001);
    values.forEach((x, i) => { const v = Math.abs(x) / max; ctx.fillStyle = x < 0 ? `rgb(${40 + v * 90},${55 + v * 95},${80 + v * 170})` : `rgb(${70 + v * 180},${50 + v * 90},${55 + v * 110})`; ctx.fillRect((i % cols) * 4, Math.floor(i / cols) * 4, 3.5, 3.5); });
  }, [values]);
  return <canvas ref={ref} className="heatmap" aria-label="高次元成分ヒートマップ。青が負、赤が正。数値は下の成分表示で確認できます。" />;
}

function DatabaseIcon() { return <span className="mono">∅</span>; }

function Prediction({ id, stage, token, step, level, onError }: { id: string; stage: string; token: number; step: number; level: number; onError: (s: string) => void }) {
  const [result, setResult] = useState<any>(null); const [busy, setBusy] = useState(false);
  useEffect(() => { setResult(null); }, [id, stage, token]);
  return <><p className="subtle-note">Logit Lensは、保存した途中層のResidualに最終NormとLM Headを適用した読み出しです。途中層でモデルが実際に出した確率とは異なります。</p><button className="secondary compact" disabled={level < 2 || busy} onClick={async () => { setBusy(true); try { setResult(await api(`traces/${id}/lens`, 'POST', { stage, token, step })); } catch (e) { onError(String(e)); } finally { setBusy(false); } }}>{busy ? 'モデルを読み込んで計算中…' : 'Logit Lensを計算'}</button>{level < 2 && <p className="subtle-note">Level 2以上のResidualが必要です。</p>}{result?.candidates.map((c: any) => <div className="prediction-row" key={c.id}><span>{c.text || c.raw}</span><small>ID {c.id}</small><b>{(c.probability * 100).toFixed(2)}%</b></div>)}</>;
}
