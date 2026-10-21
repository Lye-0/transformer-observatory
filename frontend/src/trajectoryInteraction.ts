import { PerspectiveCamera, Vector3 } from 'three';

export type Observation = { token: number; stage: number };

// OrbitControls measures rotation in radians per viewport height. Reduce that
// angle near the data so the same drag remains useful for close inspection.
export function rotationSpeed(distance: number, radius: number) {
  return 0.55 * Math.min(1, Math.max(0.04, distance / (radius * 2.8)));
}

export function pickObservation({
  data,
  count,
  stages,
  lastStage,
  activeStage,
  selected,
  camera,
  width,
  height,
  x,
  y,
}: {
  data: Float32Array;
  count: number;
  stages: number;
  lastStage: number;
  activeStage: number;
  selected: number[];
  camera: PerspectiveCamera;
  width: number;
  height: number;
  x: number;
  y: number;
}): Observation | null {
  if (width <= 0 || height <= 0 || x < 0 || x > width || y < 0 || y > height) return null;
  const point = new Vector3();
  let best: Observation | null = null;
  let bestDistance = Infinity;
  let bestDepth = Infinity;
  let bestPriority = -1;
  const selectedTokens = new Set(selected);
  for (let token = 0; token < count; token++) {
    for (let stage = 0; stage <= lastStage; stage++) {
      point.fromArray(data, (token * stages + stage) * 3).project(camera);
      if (point.z < -1 || point.z > 1) continue;
      const dx = ((point.x + 1) * width) / 2 - x;
      const dy = ((1 - point.y) * height) / 2 - y;
      const distance = dx * dx + dy * dy;
      const isSelected = stage === activeStage && selectedTokens.has(token);
      const hitRadius = isSelected ? 10 : stage === activeStage ? 8 : 7;
      if (distance > hitRadius * hitRadius) continue;
      const priority = isSelected ? 2 : stage === activeStage ? 1 : 0;
      // Screen distance wins; coincident points prefer the frontmost point,
      // then the current selection so overlapping embeddings do not flicker.
      const samePixel = Math.abs(distance - bestDistance) < 0.25;
      const sameDepth = Math.abs(point.z - bestDepth) < 1e-6;
      if (
        (!samePixel && distance < bestDistance) ||
        (samePixel && ((!sameDepth && point.z < bestDepth) || (sameDepth && priority > bestPriority)))
      ) {
        best = { token, stage };
        bestDistance = distance;
        bestDepth = point.z;
        bestPriority = priority;
      }
    }
  }
  return best;
}
