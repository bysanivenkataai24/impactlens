// Renders a photo with its review overlays into a new image. The output is
// re-encoded from canvas pixels, so no EXIF or other metadata is carried over.

import { findingNumbers, originLabel } from './findings';
import { decodeImage } from './photos';
import { DAMAGE_LABELS, REVIEW_STATUS_LABELS, type Finding, type InspectionCase, type Photo, type ReviewDamageType } from './types';

/** Okabe-Ito based palette: distinguishable with common colour-vision deficiencies. Labels carry the type too. */
export const DAMAGE_COLOURS: Record<ReviewDamageType, string> = {
  dent: '#E69F00',
  scratch: '#56B4E9',
  crack: '#F0E442',
  glass_shatter: '#009E73',
  lamp_broken: '#CC79A7',
  tire_flat: '#D55E00',
  other_uncertain: '#FFFFFF',
};

const MAX_LONG_SIDE = 3200;

export function photoSource(photo: Photo): string | null {
  return photo.object_url ?? photo.sample_url ?? null;
}

export async function renderAnnotatedImage(
  inspection: InspectionCase,
  photo: Photo,
  { quality = 0.9 }: { quality?: number } = {},
): Promise<Blob> {
  const source = photoSource(photo);
  if (!source) throw new Error('The photo bytes are not available. Re-attach the photo first.');
  const image = await decodeImage(source);
  const scale = Math.min(1, MAX_LONG_SIDE / Math.max(photo.width, photo.height));
  const width = Math.round(photo.width * scale);
  const height = Math.round(photo.height * scale);
  const unit = Math.max(width, height) / 1000;
  const footerHeight = Math.round(70 * unit);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height + footerHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas is not available in this browser.');
  ctx.drawImage(image, 0, 0, width, height);

  const findings = inspection.findings.filter((f) => f.photo_id === photo.id);
  const numbers = findingNumbers(inspection.findings);
  for (const finding of findings) drawFinding(ctx, finding, numbers.get(finding.id) ?? 0, width, height, unit);

  // Footer: provenance and status, so the image cannot be mistaken for a decision.
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, height, width, footerHeight);
  ctx.fillStyle = '#F9FAFB';
  ctx.textBaseline = 'middle';
  ctx.font = `${Math.round(18 * unit)}px system-ui, sans-serif`;
  const model = photo.model_run ? `model ${photo.model_run.model_id} @ ${photo.model_run.model_revision}` : 'no model run';
  const first =
    inspection.mode === 'illustrative_demo'
      ? 'ILLUSTRATIVE DEMO - drawing and sample findings, not model output'
      : `ImpactLens review - ${findings.length} finding(s) - ${model}`;
  ctx.fillText(first, 12 * unit, height + footerHeight * 0.32, width - 24 * unit);
  ctx.font = `${Math.round(15 * unit)}px system-ui, sans-serif`;
  ctx.fillText(
    `${photo.file_name} - sha256 ${photo.sha256.slice(0, 16)}... - decision support only, see report for review status`,
    12 * unit,
    height + footerHeight * 0.72,
    width - 24 * unit,
  );

  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Could not encode the image.'))), 'image/jpeg', quality),
  );
}

function drawFinding(ctx: CanvasRenderingContext2D, finding: Finding, number: number, width: number, height: number, unit: number) {
  const colour = DAMAGE_COLOURS[finding.damage_type];
  const rejected = finding.review_status === 'rejected';
  ctx.save();
  ctx.beginPath();
  finding.region.polygon.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x * width, y * height) : ctx.lineTo(x * width, y * height)));
  ctx.closePath();
  if (!rejected) {
    ctx.fillStyle = `${colour}33`;
    ctx.fill();
  }
  ctx.setLineDash(rejected ? [8 * unit, 6 * unit] : []);
  ctx.lineWidth = 5 * unit;
  ctx.strokeStyle = '#000000';
  ctx.stroke();
  ctx.lineWidth = 2.5 * unit;
  ctx.strokeStyle = rejected ? '#9CA3AF' : colour;
  ctx.stroke();
  ctx.restore();

  const score = finding.original ? ` ${finding.original.score.toFixed(2)}` : '';
  const label = `#${number} ${DAMAGE_LABELS[finding.damage_type]}${score} - ${originLabel(finding)} - ${REVIEW_STATUS_LABELS[finding.review_status]}`;
  ctx.font = `600 ${Math.round(16 * unit)}px system-ui, sans-serif`;
  const textWidth = Math.min(ctx.measureText(label).width, width * 0.9);
  const boxHeight = 24 * unit;
  let x = finding.region.bbox[0] * width;
  let y = finding.region.bbox[1] * height - boxHeight - 2 * unit;
  if (y < 0) y = finding.region.bbox[3] * height + 2 * unit;
  if (y + boxHeight > height) y = Math.max(0, height - boxHeight);
  x = Math.min(Math.max(0, x), width - textWidth - 10 * unit);
  ctx.fillStyle = 'rgba(17, 24, 39, 0.85)';
  ctx.fillRect(x, y, textWidth + 10 * unit, boxHeight);
  ctx.fillStyle = rejected ? '#D1D5DB' : colour;
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x + 5 * unit, y + boxHeight / 2, textWidth);
}
