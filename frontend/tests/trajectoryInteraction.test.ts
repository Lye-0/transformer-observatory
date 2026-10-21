import assert from 'node:assert/strict';
import test from 'node:test';
import { PerspectiveCamera, Vector3 } from 'three';
import { pickObservation, rotationSpeed } from '../src/trajectoryInteraction.ts';

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

test('close-up rotation is gentler, bounded and independent of data units', () => {
  const wide = rotationSpeed(28, 10);
  const close = rotationSpeed(2.8, 10);
  assert.ok(close <= wide / 9);
  assert.ok(rotationSpeed(0.0001, 10) > 0);
  assert.equal(rotationSpeed(1e9, 10), wide);
  for (const scale of [0.001, 10000])
    assert.ok(Math.abs(rotationSpeed(2.8 * scale, 10 * scale) - close) < 1e-12);
});
