import type { BBox, Point, Region } from './types';

export const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

const round6 = (value: number) => Math.round(value * 1e6) / 1e6;

export function bboxOfPolygon(points: Point[]): BBox {
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

export function polygonArea(points: Point[]): number {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}

/** A region from normalized polygon points, or null if it is degenerate. */
export function regionFromPolygon(points: Point[]): Region | null {
  const cleaned = points.map(([x, y]) => [round6(clamp01(x)), round6(clamp01(y))] as Point);
  if (cleaned.length < 3 || polygonArea(cleaned) < 1e-6) return null;
  return { bbox: bboxOfPolygon(cleaned), polygon: cleaned };
}

/** A rectangular region from percentages of the image (keyboard-friendly entry). */
export function regionFromPercentRect(left: number, top: number, width: number, height: number): Region | null {
  if (![left, top, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
  const x0 = clamp01(left / 100);
  const y0 = clamp01(top / 100);
  const x1 = clamp01((left + width) / 100);
  const y1 = clamp01((top + height) / 100);
  return regionFromPolygon([
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ]);
}

/** Normalized point under a pointer event, relative to the displayed image box. */
export function normalizedPoint(clientX: number, clientY: number, rect: DOMRect): Point {
  return [clamp01((clientX - rect.left) / rect.width), clamp01((clientY - rect.top) / rect.height)];
}

/** SVG `points` attribute in image pixel space. */
export function toSvgPoints(points: Point[], width: number, height: number): string {
  return points.map(([x, y]) => `${(x * width).toFixed(1)},${(y * height).toFixed(1)}`).join(' ');
}

export function regionsEqual(a: Region, b: Region): boolean {
  return (
    a.polygon.length === b.polygon.length &&
    a.polygon.every((p, i) => p[0] === b.polygon[i][0] && p[1] === b.polygon[i][1])
  );
}
