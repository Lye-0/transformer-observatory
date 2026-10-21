import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cellAt,
  colorLimit,
  componentCoordinates,
  componentDescription,
  fitPanel,
  gridColumns,
  heatColor,
} from '../src/heatmapMath.ts';

test('the same value has the same color across pins with different absolute maxima', () => {
  const first = [0, 0.25, -1],
    second = [0, 0.25, 8];
  const common = colorLimit([first, second]);
  assert.equal(common, 8);
  assert.equal(heatColor(first[1], common), heatColor(second[1], common));
  assert.notEqual(heatColor(first[1], colorLimit([first])), heatColor(second[1], colorLimit([second])));
  assert.deepEqual(first, [0, 0.25, -1], 'display normalization never rewrites captured values');
});

test('zero is neutral, signs differ, and all-zero or missing data does not break the color range', () => {
  assert.equal(heatColor(0, 1), heatColor(0, 100));
  assert.notEqual(heatColor(-1, 1), heatColor(1, 1));
  assert.equal(
    colorLimit([
      [0, 0],
      [NaN, Infinity],
    ]),
    1,
  );
  assert.equal(colorLimit([]), 1);
  assert.equal(heatColor(10, 2), heatColor(2, 2));
});

test('hit testing preserves component identity across zoom, excludes gutters and padding', () => {
  for (const cell of [1.75, 7, 12, 32]) {
    const columns = gridColumns(1024);
    assert.equal(
      cellAt(((237 % columns) + 0.4) * cell, (Math.floor(237 / columns) + 0.4) * cell, cell, columns, 1024),
      237,
    );
    assert.equal(cellAt(cell - 0.01, 0.1, cell, columns, 1024), null);
    assert.equal(cellAt(-1, 0, cell, columns, 1024), null);
  }
  assert.equal(cellAt(1, 13, 12, 2, 2), null);
  assert.equal(cellAt(24, 0, 12, 2, 4), null);
  assert.equal(cellAt(13, 13, 12, 2, 3), null);
});

test('flat indices map to the recorded row-major tensor shape, not visual grid columns', () => {
  assert.deepEqual(componentCoordinates(237, [1024]), [237]);
  assert.deepEqual(componentCoordinates(23, [2, 3, 4]), [1, 2, 3]);
  assert.deepEqual(componentCoordinates(0, []), []);
  const tensor = {
    name: 'attention_weights',
    space: 'attention',
    shape: [2, 7],
    heads: [3, 11],
    values: [],
    dimensions: 14,
  };
  assert.equal(componentDescription(9, tensor), '成分 9 · Head 11 / Key位置 2');
  assert.equal(componentDescription(9, { ...tensor, name: 'q' }), '成分 9 · 添字 [1, 2]');
});

test('panel dragging and resizing leave the title and close button reachable', () => {
  const moved = fitPanel({ x: 9999, y: -500, width: 620, height: 680 }, 1280, 900);
  assert.deepEqual(moved, { x: 652, y: 8, width: 620, height: 680 });
  const resized = fitPanel({ x: 100, y: 100, width: 9999, height: 9999 }, 390, 600);
  assert.deepEqual(resized, { x: 8, y: 8, width: 374, height: 584 });
  const small = fitPanel({ x: -10, y: 800, width: 10, height: 10 }, 1280, 900);
  assert.ok(small.width >= 320 && small.height >= 240 && small.y + small.height <= 892);
});
