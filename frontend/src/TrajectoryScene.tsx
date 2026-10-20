import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { Token } from './api';

type Props = {
  data: Float32Array; overlay?: Float32Array; count: number; stages: number; stage: number;
  tokens: Token[]; selected: number[]; growth: boolean; intensity: boolean; metrics?: Float32Array;
  arrows: boolean; lowPower: boolean; onSelect: (index: number, multi: boolean) => void;
  cameraBus?: React.RefObject<{ position: number[]; target: number[] } | null>;
};

export function tokenColor(i: number) { return `hsl(${(i * 137.508 + 238) % 360}, 72%, 70%)`; }

export default function TrajectoryScene(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const live = useRef(props); live.current = props;
  const cameraState = useRef<{ position: number[]; target: number[] } | null>(null);
  const [fps, setFps] = useState(0);
  const [error, setError] = useState('');
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const host = container.current!;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: !props.lowPower, alpha: true }); }
    catch { setError('WebGL2を初期化できません。ブラウザのハードウェアアクセラレーションを確認してください。'); return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, props.lowPower ? 1 : 2));
    renderer.setClearColor(0x080a0f, 0);
    host.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-label', `${props.count}本の実測トークン軌跡。選択は左のトークン一覧からも操作できます。`);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(43, 1, .01, 10000);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = .09;
    controls.screenSpacePanning = true;
    const data = props.data, n = props.count, stages = props.stages;
    const bounds = new THREE.Box3();
    const vector = new THREE.Vector3();
    for (let i = 0; i < data.length; i += 3) bounds.expandByPoint(vector.fromArray(data, i));
    if (props.overlay) for (let i = 0; i < props.overlay.length; i += 3) bounds.expandByPoint(vector.fromArray(props.overlay, i));
    const center = bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 1);
    const saved = cameraState.current;
    if (saved) { camera.position.fromArray(saved.position); controls.target.fromArray(saved.target); }
    else { camera.position.copy(center).add(new THREE.Vector3(radius * 1.25, radius * .65, radius * 2.4)); controls.target.copy(center); }
    camera.far = Math.max(1000, radius * 100); camera.updateProjectionMatrix();
    const group = new THREE.Group(); scene.add(group);
    const colors = new Float32Array(Math.max(0, n * (stages - 1) * 6));
    const positions = new Float32Array(colors.length);
    const color = new THREE.Color();
    for (let s = 1; s < stages; s++) for (let t = 0; t < n; t++) {
      const base = ((s - 1) * n + t) * 6;
      positions.set(data.subarray((t * stages + s - 1) * 3, (t * stages + s - 1) * 3 + 3), base);
      positions.set(data.subarray((t * stages + s) * 3, (t * stages + s) * 3 + 3), base + 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: .78, depthWrite: false });
    const lines = new THREE.LineSegments(geometry, material); group.add(lines);
    const pointPositions = new Float32Array(n * 3), pointColors = new Float32Array(n * 3);
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.BufferAttribute(pointPositions, 3));
    pointGeometry.setAttribute('color', new THREE.BufferAttribute(pointColors, 3));
    const points = new THREE.Points(pointGeometry, new THREE.PointsMaterial({ size: radius * .014, vertexColors: true, sizeAttenuation: true }));
    group.add(points);
    let overlayGeometry: THREE.BufferGeometry | undefined, overlayMaterial: THREE.LineBasicMaterial | undefined;
    if (props.overlay) {
      const otherCount = props.overlay.length / stages / 3;
      const other = new Float32Array(otherCount * (stages - 1) * 6);
      for (let s = 1; s < stages; s++) for (let t = 0; t < otherCount; t++) {
        const base = ((s - 1) * otherCount + t) * 6;
        other.set(props.overlay.subarray((t * stages + s - 1) * 3, (t * stages + s - 1) * 3 + 3), base);
        other.set(props.overlay.subarray((t * stages + s) * 3, (t * stages + s) * 3 + 3), base + 3);
      }
      overlayGeometry = new THREE.BufferGeometry(); overlayGeometry.setAttribute('position', new THREE.BufferAttribute(other, 3));
      overlayMaterial = new THREE.LineBasicMaterial({ color: 0xffc785, transparent: true, opacity: .5, depthWrite: false });
      group.add(new THREE.LineSegments(overlayGeometry, overlayMaterial));
    }
    const axes = new THREE.AxesHelper(radius * .22); axes.position.copy(center); axes.material.transparent = true; axes.material.opacity = .35; scene.add(axes);
    const arrowGroup = new THREE.Group(); scene.add(arrowGroup);
    const raycaster = new THREE.Raycaster(); raycaster.params.Line = { threshold: radius * .008 };
    let hovered: number | null = null, pointerDown = { x: 0, y: 0 };
    const pick = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      const pointer = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(lines)[0];
      return hit && hit.index !== undefined ? Math.floor(hit.index / 2) % n : null;
    };
    const move = (event: PointerEvent) => { const next = pick(event); if (next !== hovered) { hovered = next; setHover(next); } };
    const down = (e: PointerEvent) => { pointerDown = { x: e.clientX, y: e.clientY }; };
    const up = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y) > 5) return;
      const t = pick(e); if (t !== null) live.current.onSelect(t, e.ctrlKey || e.metaKey);
    };
    const focus = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
      const hit = raycaster.intersectObject(lines)[0];
      if (hit) controls.target.copy(hit.point);
    };
    renderer.domElement.addEventListener('pointermove', move);
    renderer.domElement.addEventListener('pointerdown', down);
    renderer.domElement.addEventListener('pointerup', up);
    renderer.domElement.addEventListener('dblclick', focus);
    const resize = new ResizeObserver(() => {
      camera.aspect = host.clientWidth / Math.max(host.clientHeight, 1); camera.updateProjectionMatrix();
      renderer.setSize(host.clientWidth, host.clientHeight);
    }); resize.observe(host);
    let animation = 0, frames = 0, lastFps = performance.now(), lastRender = 0, signature = '';
    const render = (now: number) => {
      animation = requestAnimationFrame(render);
      if (live.current.lowPower && now - lastRender < 32) return;
      lastRender = now;
      const p = live.current, key = `${p.stage}:${p.selected.join(',')}:${p.growth}:${p.intensity}:${p.arrows}:${hovered}`;
      if (key !== signature) {
        signature = key;
        geometry.setDrawRange(0, p.growth ? p.stage * n * 2 : positions.length / 3);
        let maxUpdate = 0;
        if (p.metrics) for (let i = 2; i < p.metrics.length; i += 7) maxUpdate = Math.max(maxUpdate, p.metrics[i]);
        for (let s = 1; s < stages; s++) for (let t = 0; t < n; t++) {
          color.set(tokenColor(t));
          const selected = p.selected.includes(t) || hovered === t;
          const brightness = p.selected.length && !selected ? .22 : 1;
          const intensity = p.intensity && p.metrics ? .2 + .8 * Math.log1p(p.metrics[(t * stages + s) * 7 + 2]) / Math.max(Math.log1p(maxUpdate), .001) : 1;
          color.multiplyScalar(brightness * intensity);
          const base = ((s - 1) * n + t) * 6;
          colors.set([color.r, color.g, color.b, color.r, color.g, color.b], base);
        }
        geometry.attributes.color.needsUpdate = true;
        for (let t = 0; t < n; t++) {
          pointPositions.set(data.subarray((t * stages + p.stage) * 3, (t * stages + p.stage) * 3 + 3), t * 3);
          color.set(tokenColor(t)); pointColors.set([color.r, color.g, color.b], t * 3);
        }
        pointGeometry.attributes.position.needsUpdate = true; pointGeometry.attributes.color.needsUpdate = true;
        for (const child of [...arrowGroup.children]) {
          arrowGroup.remove(child);
          if (child instanceof THREE.ArrowHelper) { child.line.geometry.dispose(); child.cone.geometry.dispose(); }
        }
        if (p.arrows && p.stage > 0) for (const t of p.selected) {
          const from = new THREE.Vector3().fromArray(data, (t * stages + p.stage - 1) * 3);
          const to = new THREE.Vector3().fromArray(data, (t * stages + p.stage) * 3);
          const direction = to.sub(from), length = direction.length();
          if (length > 1e-8) arrowGroup.add(new THREE.ArrowHelper(direction.normalize(), from, length, 0xf1d49c, length * .16, length * .065));
        }
      }
      if (p.cameraBus?.current && !controls.enabled) {
        camera.position.fromArray(p.cameraBus.current.position); controls.target.fromArray(p.cameraBus.current.target);
      }
      controls.update(); renderer.render(scene, camera);
      if (p.cameraBus) p.cameraBus.current = { position: camera.position.toArray(), target: controls.target.toArray() };
      frames++;
      if (now - lastFps > 1500) { setFps(Math.round(frames * 1000 / (now - lastFps))); frames = 0; lastFps = now; }
    }; animation = requestAnimationFrame(render);
    return () => {
      cameraState.current = { position: camera.position.toArray(), target: controls.target.toArray() };
      cancelAnimationFrame(animation); resize.disconnect(); controls.dispose();
      geometry.dispose(); pointGeometry.dispose(); material.dispose(); (points.material as THREE.Material).dispose();
      overlayGeometry?.dispose(); overlayMaterial?.dispose();
      axes.geometry.dispose(); (axes.material as THREE.Material).dispose();
      renderer.dispose(); renderer.domElement.remove();
    };
  }, [props.data, props.overlay, props.count, props.stages, props.lowPower]);

  return <div className="scene-wrap">
    <div ref={container} className="scene-canvas" />
    {error && <div className="scene-error" role="alert">{error}</div>}
    <div className="scene-caption"><span className="live-dot" /> MEASURED TRAJECTORIES <b>{props.count.toLocaleString()}</b><span> / 全トークン</span></div>
    <div className="scene-performance">{fps} FPS · {props.lowPower ? '軽量表示' : 'WebGL2'}{fps > 0 && fps < 20 ? ' · 軽量表示を推奨' : ''}</div>
    {hover !== null && <div className="scene-hover">#{hover} · {props.tokens[hover]?.raw}</div>}
    <div className="scene-help">ドラッグで回転 · 右ドラッグで移動 · ホイールで拡大 · ダブルクリックで注視</div>
  </div>;
}
