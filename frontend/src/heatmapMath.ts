export type TensorValue = {
  name: string;
  space: string;
  shape: number[];
  values: number[];
  dimensions: number;
  truncated?: boolean;
  heads?: number[];
};

export function colorLimit(series: readonly (readonly number[])[]) {
  let limit = 0;
  for (const values of series)
    for (const value of values) if (Number.isFinite(value)) limit = Math.max(limit, Math.abs(value));
  return limit || 1;
}

export function heatColor(value: number, limit: number) {
  if (!Number.isFinite(value)) return '#646b78';
  const strength = Math.min(1, Math.abs(value) / Math.max(limit, Number.MIN_VALUE));
  const end = value < 0 ? [111, 179, 255] : [255, 140, 121];
  const zero = [31, 39, 54];
  return `rgb(${zero.map((v, i) => Math.round(v + (end[i] - v) * strength)).join(',')})`;
}

export function gridColumns(count: number) {
  return Math.min(Math.max(count, 1), 2 ** Math.ceil(Math.log2(Math.sqrt(Math.max(count, 1)))));
}

export function cellAt(x: number, y: number, cell: number, columns: number, count: number) {
  if (x < 0 || y < 0 || cell <= 0 || x >= columns * cell) return null;
  const column = Math.floor(x / cell),
    row = Math.floor(y / cell);
  const index = row * columns + column;
  const gap = Math.min(1, cell * 0.15);
  return index < count && x % cell < cell - gap && y % cell < cell - gap ? index : null;
}

export function componentCoordinates(index: number, shape: number[]) {
  const coordinates = new Array<number>(shape.length);
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    coordinates[axis] = index % shape[axis];
    index = Math.floor(index / shape[axis]);
  }
  return coordinates;
}

export function componentDescription(index: number, tensor: TensorValue) {
  const coordinates = componentCoordinates(index, tensor.shape);
  if (tensor.name === 'attention_weights' && coordinates.length === 2)
    return `成分 ${index} · Head ${tensor.heads?.[coordinates[0]] ?? coordinates[0]} / Key位置 ${coordinates[1]}`;
  return `成分 ${index}${coordinates.length > 1 ? ` · 添字 [${coordinates.join(', ')}]` : ''}`;
}

export type PanelRect = { x: number; y: number; width: number; height: number };
export function fitPanel(rect: PanelRect, width: number, height: number): PanelRect {
  const availableWidth = Math.max(1, width - 16),
    availableHeight = Math.max(1, height - 16);
  const w = Math.min(availableWidth, Math.max(Math.min(320, availableWidth), rect.width));
  const h = Math.min(availableHeight, Math.max(Math.min(240, availableHeight), rect.height));
  return {
    x: Math.max(8, Math.min(rect.x, width - w - 8)),
    y: Math.max(8, Math.min(rect.y, height - h - 8)),
    width: w,
    height: h,
  };
}
