import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Token } from './api';
import { pickObservation, rotationSpeed, type Observation } from './trajectoryInteraction';

function markerTexture(ring = false) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const context = canvas.getContext('2d')!;
  context.beginPath();
  context.arc(32, 32, ring ? 26 : 28, 0, Math.PI * 2);
  if (ring) {
    context.strokeStyle = '#ffffff';
    context.lineWidth = 8;
    context.stroke();
  } else {
    context.fillStyle = '#ffffff';
    context.fill();
  }
  return new THREE.CanvasTexture(canvas);
}

type Props = {
  data: Float32Array;
  overlay?: Float32Array;
  count: number;
  stages: number;
  stage: number;
  tokens: Token[];
  selected: number[];
  growth: boolean;
  intensity: boolean;
  metrics?: Float32Array;
  arrows: boolean;
  lowPower: boolean;
  onSelect: (index: number, multi: boolean) => void;
  onStage?: (stage: number) => void;
  cameraBus?: React.RefObject<{ position: number[]; target: number[]; source?: string } | null>;
  measuredLabel?: string;
};

export function tokenColor(i: number) {
  return `hsl(${(i * 137.508 + 238) % 360}, 72%, 70%)`;
}

export default function TrajectoryScene(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const live = useRef(props);
  live.current = props;
  const cameraState = useRef<{ position: number[]; target: number[] } | null>(null);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState('');
  const [hover, setHover] = useState<Observation | null>(null);
  const resetView = useRef<(() => void) | null>(null);

  useEffect(() => {
    const host = container.current!;
    setHover(null);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: !props.lowPower, alpha: true });
    } catch {
      setError('WebGL2を初期化できません。ブラウザのハードウェアアクセラレーションを確認してください。');
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, props.lowPower ? 1 : 2));
    renderer.setClearColor(0x080a0f, 0);
    host.appendChild(renderer.domElement);
    renderer.domElement.setAttribute(
      'aria-label',
      `${props.count}本の${props.measuredLabel || '実測トークン軌跡'}。トークン選択からも操作できます。`,
    );
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(43, 1, 0.01, 10000);
    const controls = new OrbitControls(camera, renderer.domElement);
    const source = crypto.randomUUID();
    let applyingCamera = false;
    const publishCamera = () => {
      if (!applyingCamera && live.current.cameraBus)
        live.current.cameraBus.current = {
          position: camera.position.toArray(),
          target: controls.target.toArray(),
          source,
        };
    };
    controls.addEventListener('change', publishCamera);
    controls.enableDamping = true;
    controls.dampingFactor = 0.09;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 0.65;
    controls.zoomToCursor = true;
    const data = props.data,
      n = props.count,
      stages = props.stages;
    const bounds = new THREE.Box3();
    const vector = new THREE.Vector3();
    for (let i = 0; i < data.length; i += 3) bounds.expandByPoint(vector.fromArray(data, i));
    if (props.overlay)
      for (let i = 0; i < props.overlay.length; i += 3)
        bounds.expandByPoint(vector.fromArray(props.overlay, i));
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = bounds.getSize(new THREE.Vector3()).length() / 2 || 1;
    const homeOffset = new THREE.Vector3(radius * 1.25, radius * 0.65, radius * 2.4);
    controls.minDistance = radius * 0.002;
    controls.maxDistance = radius * 40;
    const settleControls = () => {
      // Clear residual damping before a programmatic camera change.
      controls.enableDamping = false;
      controls.update();
      controls.enableDamping = true;
    };
    resetView.current = () => {
      settleControls();
      const fit = Math.max(1, 1 / camera.aspect);
      camera.position.copy(center).addScaledVector(homeOffset, fit);
      controls.target.copy(center);
      controls.update();
      publishCamera();
    };
    const saved = cameraState.current;
    if (saved) {
      camera.position.fromArray(saved.position);
      controls.target.fromArray(saved.target);
    } else {
      camera.position.copy(center).add(homeOffset);
      controls.target.copy(center);
    }
    camera.far = radius * 100;
    camera.near = radius * 0.00001;
    camera.updateProjectionMatrix();
    const group = new THREE.Group();
    scene.add(group);
    const colors = new Float32Array(Math.max(0, n * (stages - 1) * 6));
    const positions = new Float32Array(colors.length);
    const color = new THREE.Color();
    for (let s = 1; s < stages; s++)
      for (let t = 0; t < n; t++) {
        const base = ((s - 1) * n + t) * 6;
        positions.set(data.subarray((t * stages + s - 1) * 3, (t * stages + s - 1) * 3 + 3), base);
        positions.set(data.subarray((t * stages + s) * 3, (t * stages + s) * 3 + 3), base + 3);
      }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.78,
      depthWrite: false,
    });
    const lines = new THREE.LineSegments(geometry, material);
    group.add(lines);
    const pointPositions = new Float32Array(n * 3),
      pointColors = new Float32Array(n * 3);
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.BufferAttribute(pointPositions, 3));
    pointGeometry.setAttribute('color', new THREE.BufferAttribute(pointColors, 3));
    const points = new THREE.Points(
      pointGeometry,
      new THREE.PointsMaterial({
        size: 8,
        vertexColors: true,
        sizeAttenuation: false,
        map: markerTexture(),
        transparent: true,
        alphaTest: 0.1,
        depthWrite: false,
      }),
    );
    group.add(points);
    // Stage-major markers allow growth mode to reveal exactly the same points
    // as its line segments, without dropping any measured observations.
    const observationPositions = new Float32Array(n * stages * 3);
    const observationColors = new Float32Array(observationPositions.length);
    for (let s = 0; s < stages; s++)
      for (let t = 0; t < n; t++)
        observationPositions.set(
          data.subarray((t * stages + s) * 3, (t * stages + s) * 3 + 3),
          (s * n + t) * 3,
        );
    const observationGeometry = new THREE.BufferGeometry();
    observationGeometry.setAttribute('position', new THREE.BufferAttribute(observationPositions, 3));
    observationGeometry.setAttribute('color', new THREE.BufferAttribute(observationColors, 3));
    const observationMaterial = points.material.clone();
    observationMaterial.size = 5;
    const observations = new THREE.Points(observationGeometry, observationMaterial);
    group.add(observations);
    const selectionGeometry = new THREE.BufferGeometry();
    const selectionPositions = new Float32Array(n * 3);
    selectionGeometry.setAttribute('position', new THREE.BufferAttribute(selectionPositions, 3));
    selectionGeometry.setDrawRange(0, 0);
    const selectionMaterial = new THREE.PointsMaterial({
      size: 16,
      sizeAttenuation: false,
      map: markerTexture(true),
      transparent: true,
      alphaTest: 0.1,
      depthWrite: false,
    });
    const selection = new THREE.Points(selectionGeometry, selectionMaterial);
    group.add(selection);
    let overlayGeometry: THREE.BufferGeometry | undefined,
      overlayMaterial: THREE.LineBasicMaterial | undefined;
    if (props.overlay) {
      const otherCount = props.overlay.length / stages / 3;
      const other = new Float32Array(otherCount * (stages - 1) * 6);
      for (let s = 1; s < stages; s++)
        for (let t = 0; t < otherCount; t++) {
          const base = ((s - 1) * otherCount + t) * 6;
          other.set(props.overlay.subarray((t * stages + s - 1) * 3, (t * stages + s - 1) * 3 + 3), base);
          other.set(props.overlay.subarray((t * stages + s) * 3, (t * stages + s) * 3 + 3), base + 3);
        }
      overlayGeometry = new THREE.BufferGeometry();
      overlayGeometry.setAttribute('position', new THREE.BufferAttribute(other, 3));
      overlayMaterial = new THREE.LineBasicMaterial({
        color: 0xffc785,
        transparent: true,
        opacity: 0.5,
        depthWrite: false,
      });
      group.add(new THREE.LineSegments(overlayGeometry, overlayMaterial));
    }
    const axes = new THREE.AxesHelper(radius * 0.22);
    axes.position.copy(center);
    axes.material.transparent = true;
    axes.material.opacity = 0.35;
    scene.add(axes);
    const arrowGroup = new THREE.Group();
    scene.add(arrowGroup);
    let hovered: Observation | null = null;
    let pointerDown: { x: number; y: number; id: number; dragged: boolean } | null = null;
    const pick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const p = live.current;
      camera.updateMatrixWorld();
      return pickObservation({
        data,
        count: n,
        stages,
        lastStage: p.growth ? p.stage : stages - 1,
        activeStage: p.stage,
        selected: p.selected,
        camera,
        width: rect.width,
        height: rect.height,
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      });
    };
    const clearHover = () => {
      hovered = null;
      setHover(null);
      renderer.domElement.style.cursor = pointerDown ? 'grabbing' : 'grab';
    };
    const move = (event: PointerEvent) => {
      if (pointerDown) {
        if (Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 5)
          pointerDown.dragged = true;
        clearHover();
        return;
      }
      const next = pick(event);
      if (next?.token !== hovered?.token || next?.stage !== hovered?.stage) {
        hovered = next;
        setHover(next);
      }
      renderer.domElement.style.cursor = next ? 'pointer' : 'grab';
    };
    const down = (e: PointerEvent) => {
      controls.rotateSpeed = rotationSpeed(camera.position.distanceTo(controls.target), radius);
      if (e.isPrimary) pointerDown = { x: e.clientX, y: e.clientY, id: e.pointerId, dragged: false };
      else if (pointerDown) pointerDown.dragged = true;
      renderer.domElement.style.cursor = 'grabbing';
    };
    const up = (e: PointerEvent) => {
      const start = pointerDown;
      pointerDown = null;
      renderer.domElement.style.cursor = 'grab';
      if (
        !start ||
        start.id !== e.pointerId ||
        e.button !== 0 ||
        start.dragged ||
        Math.hypot(e.clientX - start.x, e.clientY - start.y) > 5
      )
        return;
      const hit = pick(e);
      if (hit) {
        live.current.onSelect(hit.token, e.ctrlKey || e.metaKey);
        live.current.onStage?.(hit.stage);
        hovered = hit;
        setHover(hit);
        renderer.domElement.style.cursor = 'pointer';
      }
    };
    const cancel = () => {
      pointerDown = null;
      clearHover();
    };
    const focus = (event: MouseEvent) => {
      const hit = pick(event);
      if (!hit) return;
      settleControls();
      const target = new THREE.Vector3().fromArray(data, (hit.token * stages + hit.stage) * 3);
      camera.position.add(target.clone().sub(controls.target));
      controls.target.copy(target);
      controls.update();
      publishCamera();
      clearHover();
    };
    renderer.domElement.style.cursor = 'grab';
    renderer.domElement.addEventListener('pointermove', move);
    renderer.domElement.addEventListener('pointerdown', down);
    renderer.domElement.addEventListener('pointerup', up);
    renderer.domElement.addEventListener('pointercancel', cancel);
    renderer.domElement.addEventListener('pointerleave', clearHover);
    renderer.domElement.addEventListener('wheel', clearHover);
    renderer.domElement.addEventListener('dblclick', focus);
    let firstResize = true;
    const resize = new ResizeObserver(() => {
      camera.aspect = host.clientWidth / Math.max(host.clientHeight, 1);
      camera.updateProjectionMatrix();
      renderer.setSize(host.clientWidth, host.clientHeight);
      if (firstResize && !saved) resetView.current?.();
      firstResize = false;
    });
    resize.observe(host);
    let animation = 0,
      frames = 0,
      lastFps = performance.now(),
      lastRender = 0,
      signature = '';
    let lastMetrics: Float32Array | undefined;
    let lastCameraMessage: unknown;
    const render = (now: number) => {
      animation = requestAnimationFrame(render);
      if (live.current.lowPower && now - lastRender < 32) return;
      lastRender = now;
      const p = live.current,
        key = `${p.stage}:${p.selected.join(',')}:${p.growth}:${p.intensity}:${p.arrows}:${hovered?.token}`;
      if (lastMetrics !== p.metrics) {
        signature = '';
        lastMetrics = p.metrics;
      }
      if (key !== signature) {
        signature = key;
        const selectedTokens = new Set(p.selected);
        geometry.setDrawRange(0, p.growth ? p.stage * n * 2 : positions.length / 3);
        observationGeometry.setDrawRange(0, (p.growth ? p.stage + 1 : stages) * n);
        if (overlayGeometry) {
          const otherCount = props.overlay!.length / stages / 3;
          overlayGeometry.setDrawRange(
            0,
            p.growth ? p.stage * otherCount * 2 : overlayGeometry.attributes.position.count,
          );
        }
        let maxUpdate = 0;
        if (p.metrics)
          for (let i = 2; i < p.metrics.length; i += 7) maxUpdate = Math.max(maxUpdate, p.metrics[i]);
        for (let s = 1; s < stages; s++)
          for (let t = 0; t < n; t++) {
            color.set(tokenColor(t));
            const selected = selectedTokens.has(t) || hovered?.token === t;
            const brightness = p.selected.length && !selected ? 0.22 : 1;
            const intensity =
              p.intensity && p.metrics
                ? 0.2 +
                  (0.8 * Math.log1p(p.metrics[(t * stages + s) * 7 + 2])) /
                    Math.max(Math.log1p(maxUpdate), 0.001)
                : 1;
            color.multiplyScalar(brightness * intensity);
            const base = ((s - 1) * n + t) * 6;
            colors.set([color.r, color.g, color.b, color.r, color.g, color.b], base);
          }
        geometry.attributes.color.needsUpdate = true;
        for (let s = 0; s < stages; s++)
          for (let t = 0; t < n; t++) {
            color.set(tokenColor(t));
            if (p.selected.length && !selectedTokens.has(t) && hovered?.token !== t)
              color.multiplyScalar(0.45);
            observationColors.set([color.r, color.g, color.b], (s * n + t) * 3);
          }
        observationGeometry.attributes.color.needsUpdate = true;
        let selectedCount = 0;
        for (let t = 0; t < n; t++) {
          pointPositions.set(
            data.subarray((t * stages + p.stage) * 3, (t * stages + p.stage) * 3 + 3),
            t * 3,
          );
          color.set(tokenColor(t));
          pointColors.set([color.r, color.g, color.b], t * 3);
          if (selectedTokens.has(t)) {
            selectionPositions.set(pointPositions.subarray(t * 3, t * 3 + 3), selectedCount * 3);
            selectedCount++;
          }
        }
        pointGeometry.attributes.position.needsUpdate = true;
        pointGeometry.attributes.color.needsUpdate = true;
        selectionGeometry.setDrawRange(0, selectedCount);
        selectionGeometry.attributes.position.needsUpdate = true;
        // Positions move as the stage changes; cached bounds must follow them.
        pointGeometry.computeBoundingSphere();
        selectionGeometry.computeBoundingSphere();
        for (const child of [...arrowGroup.children]) {
          arrowGroup.remove(child);
          if (child instanceof THREE.ArrowHelper) {
            child.line.geometry.dispose();
            child.cone.geometry.dispose();
            (child.line.material as THREE.Material).dispose();
            (child.cone.material as THREE.Material).dispose();
          }
        }
        if (p.arrows && p.stage > 0)
          for (const t of p.selected) {
            const from = new THREE.Vector3().fromArray(data, (t * stages + p.stage - 1) * 3);
            const to = new THREE.Vector3().fromArray(data, (t * stages + p.stage) * 3);
            const direction = to.sub(from),
              length = direction.length();
            if (length > 1e-8)
              arrowGroup.add(
                new THREE.ArrowHelper(
                  direction.normalize(),
                  from,
                  length,
                  0xf1d49c,
                  length * 0.16,
                  length * 0.065,
                ),
              );
          }
      }
      controls.rotateSpeed = rotationSpeed(camera.position.distanceTo(controls.target), radius);
      if (
        p.cameraBus?.current &&
        p.cameraBus.current.source !== source &&
        p.cameraBus.current !== lastCameraMessage
      ) {
        applyingCamera = true;
        settleControls();
        camera.position.fromArray(p.cameraBus.current.position);
        controls.target.fromArray(p.cameraBus.current.target);
        controls.update();
        lastCameraMessage = p.cameraBus.current;
        applyingCamera = false;
      } else controls.update();
      renderer.render(scene, camera);
      frames++;
      if (now - lastFps > 1500) {
        setFps(Math.round((frames * 1000) / (now - lastFps)));
        frames = 0;
        lastFps = now;
      }
    };
    animation = requestAnimationFrame(render);
    return () => {
      cameraState.current = { position: camera.position.toArray(), target: controls.target.toArray() };
      cancelAnimationFrame(animation);
      resize.disconnect();
      resetView.current = null;
      renderer.domElement.removeEventListener('pointermove', move);
      renderer.domElement.removeEventListener('pointerdown', down);
      renderer.domElement.removeEventListener('pointerup', up);
      renderer.domElement.removeEventListener('pointercancel', cancel);
      renderer.domElement.removeEventListener('pointerleave', clearHover);
      renderer.domElement.removeEventListener('wheel', clearHover);
      renderer.domElement.removeEventListener('dblclick', focus);
      controls.dispose();
      geometry.dispose();
      pointGeometry.dispose();
      observationGeometry.dispose();
      selectionGeometry.dispose();
      material.dispose();
      points.material.map?.dispose();
      points.material.dispose();
      observationMaterial.dispose();
      selectionMaterial.map?.dispose();
      selectionMaterial.dispose();
      overlayGeometry?.dispose();
      overlayMaterial?.dispose();
      for (const child of arrowGroup.children)
        if (child instanceof THREE.ArrowHelper) {
          child.line.geometry.dispose();
          child.cone.geometry.dispose();
          (child.line.material as THREE.Material).dispose();
          (child.cone.material as THREE.Material).dispose();
        }
      axes.geometry.dispose();
      (axes.material as THREE.Material).dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [props.data, props.overlay, props.count, props.stages, props.lowPower]);

  return (
    <div className="scene-wrap">
      <div ref={container} className="scene-canvas" />
      {error && (
        <div className="scene-error" role="alert">
          {error}
        </div>
      )}
      <div className="scene-caption">
        <span className="live-dot" /> {props.measuredLabel || 'MEASURED TRAJECTORIES'}{' '}
        <b>{props.count.toLocaleString()}</b>
        <span> / 全トークン</span>
      </div>
      <div className="scene-performance">
        {fps} FPS · {props.lowPower ? '軽量表示' : 'WebGL2'}
        {fps > 0 && fps < 20 ? ' · 軽量表示を推奨' : ''}
      </div>
      <div className="scene-tools">
        <button className="button secondary" onClick={() => resetView.current?.()}>
          全体を表示
        </button>
      </div>
      {hover !== null && (
        <div className="scene-hover">
          #{hover.token} · {props.tokens[hover.token]?.raw} · 観測点 {hover.stage + 1}/{props.stages}
        </div>
      )}
      <div className="scene-help">
        <div className="scene-legend">
          <span>
            <i className="observation-dot" />
            観測点（クリック可）
          </span>
          <span>
            <i className="current-dot" />
            現在の段階
          </span>
          <span>
            <i className="selected-dot" />
            選択中
          </span>
        </div>
        ドラッグで回転 · 右ドラッグで移動 · ホイールで拡大 · ダブルクリックで注視
      </div>
    </div>
  );
}
