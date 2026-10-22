import assert from 'node:assert/strict';
import test from 'node:test';
import { PerspectiveCamera, Vector3 } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { focusedCameraPosition, navigationTarget, pickObservation } from '../src/trajectoryInteraction.ts';

function setup(data: number[], count = 1, stages = data.length / 3 / count) {
  const camera = new PerspectiveCamera(43, 800 / 600, 0.01, 1000);
  camera.position.set(0, 0, 10);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  return {
    data: new Float32Array(data),
    count,
    stages,
    lastStage: stages - 1,
    activeStage: stages - 1,
    selected: [0],
    camera,
    width: 800,
    height: 600,
    x: 400,
    y: 300,
  };
}

test('clicking an intermediate observation selects its exact token and stage', () => {
  const options = setup([-2, 0, 0, 0, 0, 0, 2, 0, 0, -2, 2, 0, 0, 2, 0, 2, 2, 0], 2);
  assert.deepEqual(pickObservation(options), { token: 0, stage: 1 });
  const point = new Vector3(0, 2, 0).project(options.camera);
  options.y = (1 - point.y) * 300;
  assert.deepEqual(pickObservation(options), { token: 1, stage: 1 });
});

test('line interiors and empty space do not create invisible click targets', () => {
  assert.equal(pickObservation(setup([-2, 0, 0, 2, 0, 0])), null);
  assert.equal(pickObservation({ ...setup([0, 0, 0]), x: 430 }), null);
  assert.equal(pickObservation({ ...setup([0, 0, 0]), width: 0 }), null);
  assert.equal(pickObservation({ ...setup([0, 0, 0]), x: -1 }), null);
});

test('hit area stays in screen pixels across zoom, viewport size and data units', () => {
  for (const scale of [0.001, 1, 10000]) {
    for (const distance of [0.1, 10, 100]) {
      const options = setup([0, 0, 0]);
      options.camera.near = scale * 0.00001;
      options.camera.far = scale * 1000;
      options.camera.position.z = scale * distance;
      options.camera.updateProjectionMatrix();
      options.camera.updateMatrixWorld();
      for (const width of [400, 800, 1600]) {
        options.width = width;
        options.camera.aspect = width / options.height;
        options.camera.updateProjectionMatrix();
        assert.deepEqual(pickObservation({ ...options, x: width / 2 + 9 }), { token: 0, stage: 0 });
        assert.equal(pickObservation({ ...options, x: width / 2 + 11 }), null);
      }
    }
  }
});

test('growth mode excludes future observations but includes embedding at stage zero', () => {
  const options = setup([-2, 0, 0, 0, 0, 0, 2, 0, 0]);
  assert.equal(pickObservation({ ...options, lastStage: 0 }), null);
  assert.deepEqual(pickObservation({ ...options, lastStage: 1 }), { token: 0, stage: 1 });
  const embedding = setup([0, 0, 0, 2, 0, 0]);
  assert.deepEqual(pickObservation({ ...embedding, lastStage: 0, activeStage: 0 }), { token: 0, stage: 0 });
});

test('points behind the camera are ignored, frontmost coincident projections win', () => {
  assert.equal(pickObservation(setup([0, 0, 11])), null);
  assert.deepEqual(pickObservation(setup([0, 0, 0, 0, 0, 2], 2)), { token: 1, stage: 0 });
});

test('coincident embeddings prefer the active selection deterministically', () => {
  const options = setup([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 2, 2);
  assert.deepEqual(pickObservation({ ...options, selected: [1] }), { token: 1, stage: 1 });
});

function anchorOptions(options: ReturnType<typeof setup>) {
  return { ...options, data: [options.data] };
}

test('a stale pivot can be replaced without moving the camera or changing its view', () => {
  const options = setup([0.1, 0.1, 9, -0.1, -0.1, 9]);
  const camera = options.camera;
  const before = camera.quaternion.clone();
  const position = camera.position.clone();
  const target = navigationTarget(anchorOptions(options))!;
  assert.ok(Math.abs(camera.position.distanceTo(target) - 1) < 1e-10);
  camera.lookAt(target);
  assert.ok(camera.position.distanceTo(position) < 1e-10);
  assert.ok(camera.quaternion.angleTo(before) < 1e-7);
});

test('navigation depth follows visible nearby data, excluding offscreen outliers and future stages', () => {
  const options = setup([0, 0, 8, 0, 0, 9, 10000, 0, -100]);
  const visible = navigationTarget({ ...anchorOptions(options), lastStage: 0 })!;
  assert.ok(Math.abs(visible.z - 8) < 1e-6);
  const current = navigationTarget(anchorOptions(options))!;
  assert.ok(Math.abs(current.z - 9) < 1e-6);
  assert.equal(navigationTarget(anchorOptions(setup([10000, 0, -100, 0, 0, 11]))), null);
});

test('an isolated foreground marker wins under the pointer; empty space uses local median depth', () => {
  const options = setup([0, 0, 9, -0.2, 0, 5, 0.2, 0, 5, 0, 0.2, 5], 4, 1);
  assert.ok(Math.abs(navigationTarget(anchorOptions(options))!.z - 9) < 1e-6);
  assert.ok(Math.abs(navigationTarget({ ...anchorOptions(options), x: 500, y: 400 })!.z - 5) < 1e-6);
});

test('navigation depth includes comparison coordinates and is independent of data units', () => {
  const options = setup([10000, 0, -100]);
  const other = new Float32Array([0, 0, 9]);
  assert.ok(
    Math.abs(navigationTarget({ ...anchorOptions(options), data: [options.data, other] })!.z - 9) < 1e-6,
  );
  for (const scale of [0.001, 1, 10000]) {
    const scaled = setup([0, 0, 0]);
    scaled.camera.position.z = scale * 10;
    scaled.camera.near = scale * 0.01;
    scaled.camera.far = scale * 1000;
    scaled.camera.updateProjectionMatrix();
    const target = navigationTarget(anchorOptions(scaled))!;
    assert.ok(target.length() < scale * 1e-8);
  }
});

test('a visible line provides depth when both observations are outside the viewport', () => {
  const options = setup([-100, 0, 8, 100, 0, 8]);
  assert.ok(Math.abs(navigationTarget(anchorOptions(options))!.z - 8) < 1e-6);
  assert.equal(navigationTarget({ ...anchorOptions(options), lastStage: 0 }), null);
  assert.equal(pickObservation(options), null, 'navigation does not make line interiors clickable');
});

test('line depth is perspective-correct and clips endpoints behind the camera', () => {
  // The line crosses the central viewing ray at world z=7 (depth=3).
  const options = setup([-10, 0, 9, 10, 0, 5]);
  assert.ok(Math.abs(navigationTarget(anchorOptions(options))!.z - 7) < 1e-5);
  // One endpoint is behind the camera, but the visible part crosses z=8.
  const crossing = setup([-10, 0, 12, 10, 0, 4]);
  assert.ok(Math.abs(navigationTarget(anchorOptions(crossing))!.z - 8) < 1e-5);
});

test('focus preserves apparent zoom and orientation even when the old pivot is far away', () => {
  const { camera } = setup([0, 0, 0]);
  const point = new Vector3(0.2, 0.1, 9);
  const adjacent = point.clone().add(new Vector3(0.02, 0.01, 0));
  const sizeBefore = point.clone().project(camera).distanceTo(adjacent.clone().project(camera));
  const orientation = camera.quaternion.clone();
  camera.position.copy(focusedCameraPosition(camera, point));
  camera.lookAt(point);
  camera.updateMatrixWorld();
  const projected = point.clone().project(camera);
  const sizeAfter = projected.distanceTo(adjacent.clone().project(camera));
  assert.ok(Math.abs(projected.x) < 1e-8 && Math.abs(projected.y) < 1e-8);
  assert.ok(Math.abs(sizeBefore - sizeAfter) < 1e-8);
  assert.ok(camera.quaternion.angleTo(orientation) < 1e-7);
});

// OrbitControls uses browser pointer events; this small surface lets the real
// control implementation consume identical drags at different update rates.
class ControlSurface extends EventTarget {
  style = { touchAction: '' };
  clientHeight = 600;
  clientWidth = 800;
  getRootNode() {
    return this;
  }
  setPointerCapture() {}
  releasePointerCapture() {}
  pointer(type: string, x: number, y: number) {
    const event = new Event(type);
    Object.assign(event, { pointerId: 1, pointerType: 'mouse', button: 0, clientX: x, clientY: y });
    this.dispatchEvent(event);
  }
}

test('the same close view rotates equally after short or long pivot distances, regardless of global extent', () => {
  const results: number[] = [];
  for (const oldDistance of [0.01, 1, 1000]) {
    for (const extraFrames of [0, 3, 12]) {
      const options = setup([0.1, 0, 9, -0.1, 0, 9, 10000, 0, -100]);
      const surface = new ControlSurface();
      const controls = new OrbitControls(options.camera, surface as unknown as HTMLElement);
      controls.target.set(0, 0, 10 - oldDistance);
      controls.rotateSpeed = 0.55;
      controls.enableDamping = false;
      controls.update();
      controls.target.copy(navigationTarget(anchorOptions(options))!);
      controls.update();
      const sample = new Vector3(0.1, 0, 9);
      const before = sample.clone().project(options.camera);
      surface.pointer('pointerdown', 400, 300);
      for (let step = 1; step <= 6; step++) {
        surface.pointer('pointermove', 400 + step * 10, 300);
        for (let frame = 0; frame < extraFrames; frame++) controls.update();
      }
      surface.pointer('pointerup', 460, 300);
      options.camera.updateMatrixWorld();
      const after = sample.clone().project(options.camera);
      results.push((Math.abs(after.x - before.x) * options.width) / 2);
      const stopped = options.camera.position.clone();
      for (let frame = 0; frame < 180; frame++) controls.update();
      assert.ok(options.camera.position.distanceTo(stopped) < 1e-8, 'rotation stops at release');
      controls.dispose();
    }
  }
  assert.ok(Math.max(...results) - Math.min(...results) < 1e-6, 'approach history does not change response');
  assert.ok(results[0] > 0.1 && results[0] < 60, 'local geometry neither freezes nor flies away');
});
