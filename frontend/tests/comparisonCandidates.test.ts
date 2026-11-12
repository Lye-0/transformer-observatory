import assert from 'node:assert/strict';
import test from 'node:test';
import { canCompare } from '../src/comparisonCandidates.ts';

const source = {
  id: 'source',
  level: 2,
  model: { fingerprint: 'weights-and-tokenizer' },
  stages: ['embedding', 'layer.0.mlp'],
};
const candidate = {
  id: 'other',
  level: 4,
  status: 'completed',
  available: true,
  model_fingerprint: 'weights-and-tokenizer',
  stages: ['embedding', 'layer.0.mlp'],
};

test('comparison requires matching weights, observations, residuals and readable completed data', () => {
  assert.equal(canCompare(source, candidate), true);
  for (const change of [
    { id: 'source' },
    { model_fingerprint: 'other weights' },
    { stages: ['embedding'] },
    { stages: undefined },
    { level: 1 },
    { status: 'incomplete' },
    { available: false },
  ]) {
    assert.equal(canCompare(source, { ...candidate, ...change }), false);
  }
  assert.equal(canCompare({ ...source, level: 1 }, candidate), false);
  assert.equal(canCompare(source, { ...candidate, stages: ['layer.0.mlp', 'embedding'] }), false);
});
