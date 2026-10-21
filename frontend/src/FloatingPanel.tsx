import { useId, useLayoutEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { Grip, GripHorizontal, RotateCcw, X } from 'lucide-react';
import { fitPanel, type PanelRect } from './heatmapMath';

const panels = new Set<HTMLElement>();
function promote(panel: HTMLElement) {
  panels.delete(panel);
  panels.add(panel);
  for (const item of panels) item.style.zIndex = item === panel ? '81' : '80';
}

export default function FloatingPanel({
  id,
  title,
  anchor,
  fallbackFocus,
  width = 520,
  height = 560,
  onClose,
  children,
}: {
  id?: string;
  title: string;
  anchor: RefObject<HTMLElement | null>;
  fallbackFocus?: RefObject<HTMLElement | null>;
  width?: number;
  height?: number;
  onClose: () => void;
  children: ReactNode;
}) {
  const label = useId();
  const panel = useRef<HTMLElement>(null),
    close = useRef<HTMLButtonElement>(null);
  const liveClose = useRef(onClose);
  liveClose.current = onClose;
  const initial = () => {
    const bounds = anchor.current?.getBoundingClientRect();
    return fitPanel(
      { x: (bounds?.right ?? window.innerWidth) - width - 12, y: (bounds?.top ?? 90) + 12, width, height },
      window.innerWidth,
      window.innerHeight,
    );
  };
  const [rect, setRect] = useState(initial);
  const gesture = useRef<{ pointer: number; x: number; y: number; start: PanelRect; resize: boolean } | null>(
    null,
  );
  useLayoutEffect(() => {
    const element = panel.current!,
      previous = document.activeElement as HTMLElement | null;
    promote(element);
    close.current?.focus({ preventScroll: true });
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented && [...panels].at(-1) === element) {
        event.preventDefault();
        event.stopPropagation();
        liveClose.current();
      }
    };
    const resize = () => setRect((old) => fitPanel(old, window.innerWidth, window.innerHeight));
    window.addEventListener('keydown', escape);
    window.addEventListener('resize', resize);
    return () => {
      const restore = element.contains(document.activeElement);
      panels.delete(element);
      const next = [...panels].at(-1);
      if (next) promote(next);
      window.removeEventListener('keydown', escape);
      window.removeEventListener('resize', resize);
      if (restore)
        (previous?.isConnected ? previous : fallbackFocus?.current)?.focus({ preventScroll: true });
    };
  }, []);
  const pointer = (event: React.PointerEvent<HTMLButtonElement>, resize: boolean) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    gesture.current = { pointer: event.pointerId, x: event.clientX, y: event.clientY, start: rect, resize };
  };
  const move = (event: React.PointerEvent<HTMLButtonElement>) => {
    const g = gesture.current;
    if (!g || g.pointer !== event.pointerId) return;
    const dx = event.clientX - g.x,
      dy = event.clientY - g.y;
    setRect(
      fitPanel(
        g.resize
          ? { ...g.start, width: g.start.width + dx, height: g.start.height + dy }
          : { ...g.start, x: g.start.x + dx, y: g.start.y + dy },
        window.innerWidth,
        window.innerHeight,
      ),
    );
  };
  const stop = () => {
    gesture.current = null;
  };
  const keyboard = (event: React.KeyboardEvent, resize: boolean) => {
    const step = event.shiftKey ? 80 : 20;
    const dx = event.key === 'ArrowRight' ? step : event.key === 'ArrowLeft' ? -step : 0;
    const dy = event.key === 'ArrowDown' ? step : event.key === 'ArrowUp' ? -step : 0;
    if (!dx && !dy) return;
    event.preventDefault();
    setRect((old) =>
      fitPanel(
        resize
          ? { ...old, width: old.width + dx, height: old.height + dy }
          : { ...old, x: old.x + dx, y: old.y + dy },
        window.innerWidth,
        window.innerHeight,
      ),
    );
  };
  return createPortal(
    <section
      id={id}
      ref={panel}
      role="dialog"
      aria-modal="false"
      aria-labelledby={label}
      className="floating-panel"
      style={{ left: rect.x, top: rect.y, width: rect.width, height: rect.height }}
      onPointerDownCapture={() => panel.current && promote(panel.current)}
      onFocusCapture={() => panel.current && promote(panel.current)}
    >
      <header className="floating-panel-header">
        <button
          className="floating-panel-handle"
          onPointerDown={(e) => pointer(e, false)}
          onPointerMove={move}
          onPointerUp={stop}
          onPointerCancel={stop}
          onKeyDown={(e) => keyboard(e, false)}
          aria-label={`${title}を移動（矢印キーでも操作できます）`}
          title="ドラッグまたは矢印キーで移動"
        >
          <GripHorizontal size={16} />
          <span id={label}>{title}</span>
        </button>
        <button
          className="icon-button"
          aria-label="位置とサイズを戻す"
          title="位置とサイズを戻す"
          onClick={() => setRect(initial())}
        >
          <RotateCcw size={15} />
        </button>
        <button
          ref={close}
          className="icon-button"
          aria-label={`${title}を閉じる`}
          title="閉じる（Esc）"
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </header>
      <div className="floating-panel-body">{children}</div>
      <button
        className="floating-panel-resize"
        aria-label={`${title}のサイズを変更（矢印キーでも操作できます）`}
        title="ドラッグまたは矢印キーでサイズ変更"
        onPointerDown={(e) => pointer(e, true)}
        onPointerMove={move}
        onPointerUp={stop}
        onPointerCancel={stop}
        onKeyDown={(e) => keyboard(e, true)}
      >
        <Grip size={16} />
      </button>
    </section>,
    document.body,
  );
}
