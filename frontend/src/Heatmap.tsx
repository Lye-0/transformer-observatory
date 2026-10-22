import { createContext, useContext, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Hand, Maximize2, Minus, Plus } from 'lucide-react';
import NumberInput from './NumberInput';
import {
  cellAt,
  colorLimit,
  componentDescription,
  gridColumns,
  heatColor,
  type TensorValue,
} from './heatmapMath';

type Props = { tensor: TensorValue; context: string; scale?: number; onExpand?: () => void };
const valueText = (value: number) => String(value);

function ColorLegend({ limit, shared }: { limit: number; shared: boolean }) {
  return (
    <div
      className="heatmap-legend"
      aria-label={`${shared ? 'ピン共通' : '表示値'}の色範囲：${-limit}から${limit}。青が負、赤が正。`}
    >
      <div>
        <span>−{limit.toPrecision(4)}</span>
        <span>0</span>
        <span>+{limit.toPrecision(4)}</span>
      </div>
      <i />
      <small>青：負 · 赤：正 · {shared ? 'ピン共通の色範囲' : '表示値の色範囲'}</small>
    </div>
  );
}

function Grid({
  tensor,
  context,
  cell,
  selected,
  onSelect,
}: {
  tensor: TensorValue;
  context: string;
  cell: number;
  selected: number | null;
  onSelect: (index: number, reveal?: boolean) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null),
    raster = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null);
  const columns = gridColumns(tensor.values.length),
    rows = Math.ceil(tensor.values.length / columns);
  const width = columns * cell,
    height = Math.max(1, rows * cell);
  const scale = useScale();
  useEffect(() => {
    const image = document.createElement('canvas'),
      dpr = Math.min(devicePixelRatio || 1, 2, 2048 / Math.max(width, height));
    image.width = Math.ceil(width * dpr);
    image.height = Math.ceil(height * dpr);
    const ctx = image.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0b0f17';
    ctx.fillRect(0, 0, width, height);
    const gap = Math.min(1, cell * 0.15);
    tensor.values.forEach((value, index) => {
      ctx.fillStyle = heatColor(value, scale);
      ctx.fillRect((index % columns) * cell, Math.floor(index / columns) * cell, cell - gap, cell - gap);
    });
    raster.current = image;
  }, [tensor.values, width, height, cell, columns, scale]);
  useEffect(() => {
    const target = canvas.current!,
      image = raster.current;
    if (!image) return;
    target.width = image.width;
    target.height = image.height;
    const ctx = target.getContext('2d')!;
    ctx.scale(target.width / width, target.height / height);
    ctx.drawImage(image, 0, 0, width, height);
    const mark = (index: number, color: string, lineWidth: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(
        (index % columns) * cell + 0.5,
        Math.floor(index / columns) * cell + 0.5,
        cell - 1,
        cell - 1,
      );
    };
    if (hover) mark(hover.index, '#c6baf4', 1);
    if (selected !== null) mark(selected, '#ffffff', Math.min(2, cell / 3));
  }, [tensor.values, width, height, cell, columns, scale, selected, hover]);
  useEffect(() => setHover(null), [tensor.values, cell]);
  useEffect(() => {
    const clear = () => setHover(null);
    window.addEventListener('scroll', clear, true);
    window.addEventListener('resize', clear);
    return () => {
      window.removeEventListener('scroll', clear, true);
      window.removeEventListener('resize', clear);
    };
  }, []);
  const indexAt = (event: React.PointerEvent) => {
    const rect = canvas.current!.getBoundingClientRect();
    return cellAt(
      ((event.clientX - rect.left) * width) / rect.width,
      ((event.clientY - rect.top) * height) / rect.height,
      cell,
      columns,
      tensor.values.length,
    );
  };
  const pressed = useRef<{ x: number; y: number; dragged: boolean } | null>(null);
  return (
    <>
      <canvas
        ref={canvas}
        className="tensor-heatmap-canvas"
        style={{ width, height }}
        tabIndex={0}
        role="img"
        aria-label={`${context}。${tensor.values.length}成分。クリックまたは矢印キーで成分を選択できます。`}
        onPointerMove={(e) => {
          if (pressed.current && Math.hypot(e.clientX - pressed.current.x, e.clientY - pressed.current.y) > 5)
            pressed.current.dragged = true;
          const index = indexAt(e);
          setHover(index === null ? null : { index, x: e.clientX, y: e.clientY });
        }}
        onPointerLeave={() => {
          if (pressed.current) pressed.current.dragged = true;
          setHover(null);
        }}
        onBlur={() => setHover(null)}
        onPointerDown={(e) => {
          if (e.button === 0) pressed.current = { x: e.clientX, y: e.clientY, dragged: false };
        }}
        onPointerCancel={() => {
          pressed.current = null;
        }}
        onPointerUp={(e) => {
          const start = pressed.current;
          pressed.current = null;
          if (
            !start ||
            start.dragged ||
            e.button !== 0 ||
            Math.hypot(e.clientX - start.x, e.clientY - start.y) > 5
          )
            return;
          const index = indexAt(e);
          if (index !== null) onSelect(index);
        }}
        onKeyDown={(e) => {
          const delta =
            e.key === 'ArrowRight'
              ? 1
              : e.key === 'ArrowLeft'
                ? -1
                : e.key === 'ArrowDown'
                  ? columns
                  : e.key === 'ArrowUp'
                    ? -columns
                    : 0;
          if (!delta && e.key !== 'Home' && e.key !== 'End') return;
          e.preventDefault();
          onSelect(
            e.key === 'Home'
              ? 0
              : e.key === 'End'
                ? tensor.values.length - 1
                : Math.max(0, Math.min(tensor.values.length - 1, (selected ?? 0) + delta)),
            true,
          );
        }}
      />
      {hover &&
        createPortal(
          <div
            role="tooltip"
            className="heatmap-tooltip"
            style={{
              left: Math.max(8, Math.min(hover.x + 14, window.innerWidth - 284)),
              top: Math.max(8, Math.min(hover.y + 14, window.innerHeight - 130)),
            }}
          >
            <small>{context}</small>
            <b>{componentDescription(hover.index, tensor)}</b>
            <strong>{valueText(tensor.values[hover.index])}</strong>
          </div>,
          document.body,
        )}
    </>
  );
}

// Scale belongs to the view, so the same grid renders identically in preview,
// detail and pinned comparisons without changing the recorded tensor values.
const Scale = createContext(1);
const useScale = () => useContext(Scale);

function Selection({ index, tensor }: { index: number | null; tensor: TensorValue }) {
  return (
    <div className="heatmap-selection" aria-live="polite" data-component-index={index ?? undefined}>
      {index === null ? (
        <span>成分をクリックすると値を保持します。</span>
      ) : (
        <>
          <span>{componentDescription(index, tensor)}</span>
          <strong>{valueText(tensor.values[index])}</strong>
        </>
      )}
    </div>
  );
}

export default function Heatmap({ tensor, context, scale, onExpand }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(224),
    [selected, setSelected] = useState<number | null>(null);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(host.current!);
    return () => observer.disconnect();
  }, []);
  const columns = gridColumns(tensor.values.length),
    rows = Math.ceil(tensor.values.length / columns);
  const cell = Math.max(0.5, Math.min(12, width / columns, 224 / Math.max(1, rows)));
  const limit = scale ?? colorLimit([tensor.values]);
  return (
    <div className="heatmap-view" ref={host}>
      <div className="heatmap-toolbar">
        <span>成分マップ</span>
        {onExpand && (
          <button className="text-button" onClick={onExpand}>
            <Maximize2 size={14} />
            拡大して見る
          </button>
        )}
      </div>
      {tensor.values.length ? (
        <Scale value={limit}>
          <div className="heatmap-preview">
            <Grid tensor={tensor} context={context} cell={cell} selected={selected} onSelect={setSelected} />
          </div>
        </Scale>
      ) : (
        <p className="subtle-note">表示できる成分がありません。</p>
      )}
      <ColorLegend limit={limit} shared={scale !== undefined} />
      <Selection index={selected} tensor={tensor} />
    </div>
  );
}

export function HeatmapDetail({ tensor, context, scale }: Props) {
  const [cell, setCell] = useState(12),
    [selected, setSelected] = useState<number | null>(null),
    [input, setInput] = useState(''),
    [pan, setPan] = useState(false);
  const [revealTick, setRevealTick] = useState(0);
  const viewport = useRef<HTMLDivElement>(null),
    previousCell = useRef(cell);
  const drag = useRef<{ id: number; x: number; y: number; left: number; top: number } | null>(null);
  const reveal = useRef(false),
    inputId = useId();
  const columns = gridColumns(tensor.values.length),
    limit = scale ?? colorLimit([tensor.values]);
  const choose = (index: number, scroll = false) => {
    reveal.current = scroll;
    setSelected(index);
    setInput(String(index));
    if (scroll) setRevealTick((old) => old + 1);
  };
  useLayoutEffect(() => {
    const element = viewport.current!;
    const ratio = cell / previousCell.current;
    if (ratio !== 1) {
      element.scrollLeft = (element.scrollLeft + element.clientWidth / 2) * ratio - element.clientWidth / 2;
      element.scrollTop = (element.scrollTop + element.clientHeight / 2) * ratio - element.clientHeight / 2;
      previousCell.current = cell;
    }
    if (reveal.current && selected !== null) {
      const x = (selected % columns) * cell,
        y = Math.floor(selected / columns) * cell;
      if (x < element.scrollLeft || x + cell > element.scrollLeft + element.clientWidth)
        element.scrollLeft = x - element.clientWidth / 2;
      if (y < element.scrollTop || y + cell > element.scrollTop + element.clientHeight)
        element.scrollTop = y - element.clientHeight / 2;
      reveal.current = false;
    }
  }, [cell, selected, columns, revealTick]);
  return (
    <div className="heatmap-detail">
      <p className="heatmap-context">
        {context}
        <br />
        <span>
          {tensor.space} · 形状 [{tensor.shape.join(' × ')}] · {tensor.dimensions.toLocaleString()}成分
        </span>
      </p>
      <div className="heatmap-detail-tools">
        <button
          className="icon-button"
          aria-label="成分マップを縮小"
          disabled={cell <= 6}
          onClick={() => setCell((old) => Math.max(6, old - 2))}
        >
          <Minus size={16} />
        </button>
        <span className="mono">{Math.round((cell / 12) * 100)}%</span>
        <button
          className="icon-button"
          aria-label="成分マップを拡大"
          disabled={cell >= 32}
          onClick={() => setCell((old) => Math.min(32, old + 2))}
        >
          <Plus size={16} />
        </button>
        <button
          className={`text-button ${pan ? 'active' : ''}`}
          aria-pressed={pan}
          onClick={() => setPan(!pan)}
        >
          <Hand size={15} />
          移動
        </button>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const index = Number(input);
            if (input.trim() && Number.isInteger(index) && index >= 0 && index < tensor.values.length)
              choose(index, true);
          }}
        >
          <label htmlFor={inputId}>成分番号</label>
          <NumberInput
            id={inputId}
            aria-label="成分番号"
            min={0}
            max={tensor.values.length - 1}
            step={1}
            value={input}
            onValueChange={setInput}
            required
          />
          <button className="text-button" type="submit">
            移動する
          </button>
        </form>
      </div>
      <Scale value={limit}>
        <div
          ref={viewport}
          className="heatmap-viewport"
          data-pan={pan}
          onPointerDownCapture={(e) => {
            if (!pan || e.button !== 0) return;
            e.preventDefault();
            e.stopPropagation();
            const element = e.currentTarget;
            element.setPointerCapture(e.pointerId);
            drag.current = {
              id: e.pointerId,
              x: e.clientX,
              y: e.clientY,
              left: element.scrollLeft,
              top: element.scrollTop,
            };
          }}
          onPointerMove={(e) => {
            const d = drag.current;
            if (!d || d.id !== e.pointerId) return;
            e.currentTarget.scrollLeft = d.left - e.clientX + d.x;
            e.currentTarget.scrollTop = d.top - e.clientY + d.y;
          }}
          onPointerUp={() => {
            drag.current = null;
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
        >
          <Grid tensor={tensor} context={context} cell={cell} selected={selected} onSelect={choose} />
        </div>
      </Scale>
      <Selection index={selected} tensor={tensor} />
      <ColorLegend limit={limit} shared={scale !== undefined} />
      <p className="subtle-note">
        成分番号順の折り返し表示・0始まり。隣接は意味的な近さを示しません。矢印キーで選択、移動モードでドラッグできます。表示対象は開いた時点の値です。
        {tensor.truncated && ` 表示は先頭${tensor.values.length.toLocaleString()}成分です。`}
      </p>
    </div>
  );
}
