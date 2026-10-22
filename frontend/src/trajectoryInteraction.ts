import { Frustum, PerspectiveCamera, Vector3 } from 'three';

export type Observation = { token: number; stage: number };

// Orbit at the depth of the visible data, rather than a stale scene-wide pivot.
// Moving the target along the viewing axis leaves the current image unchanged.
export function navigationTarget({
  data,
  stages,
  lastStage,
  camera,
  width,
  height,
  x,
  y,
}: {
  data: readonly Float32Array[];
  stages: number;
  lastStage: number;
  camera: PerspectiveCamera;
  width: number;
  height: number;
  x: number;
  y: number;
}): Vector3 | null {
  if (width <= 0 || height <= 0) return null;
  camera.updateMatrixWorld();
  const point = new Vector3();
  const neighbors: { distance: number; depth: number }[] = [];
  for (const positions of data) {
    for (let i = 0; i < positions.length / 3; i++) {
      if (i % stages > lastStage) continue;
      point.fromArray(positions, i * 3).applyMatrix4(camera.matrixWorldInverse);
      const depth = -point.z;
      if (depth <= camera.near * 4) continue;
      point.applyMatrix4(camera.projectionMatrix);
      if (Math.abs(point.x) > 1 || Math.abs(point.y) > 1 || Math.abs(point.z) > 1) continue;
      const dx = ((point.x + 1) * width) / 2 - x;
      const dy = ((1 - point.y) * height) / 2 - y;
      const distance = dx * dx + dy * dy;
      if (!Number.isFinite(distance)) continue;
      // Bound the working set, while examining every visible observation.
      let at = neighbors.length;
      while (at > 0) {
        const previous = neighbors[at - 1];
        if (distance > previous.distance || (distance === previous.distance && depth >= previous.depth))
          break;
        at--;
      }
      if (at >= 32) continue;
      neighbors.splice(at, 0, { distance, depth });
      if (neighbors.length > 32) neighbors.pop();
    }
  }
  // A marker under the pointer wins. Otherwise, a trajectory may be visible
  // even when both of its measured endpoints are outside the viewport.
  const markerDepth = neighbors[0]?.distance <= 12 ** 2 ? neighbors[0].depth : undefined;
  let segmentDepth: number | undefined;
  let segmentDistance = Infinity;
  if (markerDepth === undefined && stages > 1) {
    const frustum = new Frustum().setFromProjectionMatrix(camera.projectionMatrix);
    const start = new Vector3(),
      end = new Vector3(),
      a = new Vector3(),
      b = new Vector3();
    for (const positions of data) {
      for (let i = 0; i < positions.length / 3; i++) {
        if (i % stages === 0 || i % stages > lastStage) continue;
        start.fromArray(positions, (i - 1) * 3).applyMatrix4(camera.matrixWorldInverse);
        end.fromArray(positions, i * 3).applyMatrix4(camera.matrixWorldInverse);
        let from = 0,
          to = 1;
        // Clip in camera space before projecting, including near/far planes.
        for (const plane of frustum.planes) {
          const d0 = plane.distanceToPoint(start),
            d1 = plane.distanceToPoint(end);
          if (d0 < 0 && d1 < 0) {
            from = 1;
            to = 0;
            break;
          }
          if (d0 < 0) from = Math.max(from, d0 / (d0 - d1));
          else if (d1 < 0) to = Math.min(to, d0 / (d0 - d1));
        }
        if (from > to) continue;
        a.copy(start).lerp(end, from);
        b.copy(start).lerp(end, to);
        const depth0 = -a.z,
          depth1 = -b.z;
        a.applyMatrix4(camera.projectionMatrix);
        b.applyMatrix4(camera.projectionMatrix);
        const ax = ((a.x + 1) * width) / 2,
          ay = ((1 - a.y) * height) / 2;
        const dx = ((b.x - a.x) * width) / 2,
          dy = ((a.y - b.y) * height) / 2;
        const lengthSq = dx * dx + dy * dy;
        const t = lengthSq > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / lengthSq)) : 0;
        const distance = (ax + dx * t - x) ** 2 + (ay + dy * t - y) ** 2;
        // Depth is perspective-correct, rather than linear in screen pixels.
        const depth = 1 / ((1 - t) / depth0 + t / depth1);
        if (depth <= camera.near * 4 || !Number.isFinite(depth)) continue;
        const samePixel = Math.abs(distance - segmentDistance) < 0.25;
        if ((!samePixel && distance < segmentDistance) || (samePixel && depth < (segmentDepth ?? Infinity))) {
          segmentDistance = distance;
          segmentDepth = depth;
        }
      }
    }
  }
  if (!neighbors.length && segmentDepth === undefined) return null;
  // In empty space use a local median, rather than an isolated depth outlier.
  const depths = neighbors.map((neighbor) => neighbor.depth).sort((a, b) => a - b);
  const depth =
    markerDepth ??
    (segmentDepth !== undefined &&
    (!neighbors.length || (segmentDistance <= 64 ** 2 && segmentDistance < neighbors[0].distance))
      ? segmentDepth
      : depths[Math.floor(depths.length / 2)]);
  return camera.getWorldDirection(new Vector3()).multiplyScalar(depth).add(camera.position);
}

export function focusedCameraPosition(camera: PerspectiveCamera, point: Vector3) {
  const direction = camera.getWorldDirection(new Vector3());
  const depth = point.clone().sub(camera.position).dot(direction);
  // Recenter laterally, retaining the point's projected scale and view direction.
  return point.clone().addScaledVector(direction, -depth);
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
