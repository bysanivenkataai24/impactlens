import { regionsEqual } from './geometry';
import { newId } from './photos';
import type { Finding, ModelPrediction, ModelRun, PredictResponse, Region } from './types';

const now = () => new Date().toISOString();

export function modelRunFrom(response: PredictResponse): ModelRun {
  return {
    request_id: response.request_id,
    model_id: response.model_id,
    model_revision: response.model_revision,
    checkpoint_sha256: response.checkpoint_sha256,
    schema_version: response.schema_version,
    score_threshold: response.score_threshold,
    outcome: response.outcome,
    message: response.message,
    warnings: response.warnings,
    limitations: response.limitations,
    latency_ms: response.latency_ms,
    completed_at: now(),
  };
}

export function findingsFromResponse(photoId: string, response: PredictResponse): Finding[] {
  return response.detections.map((detection) => {
    const region: Region = { bbox: detection.bbox, polygon: detection.polygon };
    const original: ModelPrediction = Object.freeze({
      detection_id: detection.id,
      damage_type: detection.damage_type,
      score: detection.score,
      region,
      flags: detection.flags,
      request_id: response.request_id,
      model_id: response.model_id,
      model_revision: response.model_revision,
      checkpoint_sha256: response.checkpoint_sha256,
      score_threshold: response.score_threshold,
      schema_version: response.schema_version,
    });
    const timestamp = now();
    return {
      id: newId('finding'),
      photo_id: photoId,
      origin: 'model_output',
      original,
      damage_type: detection.damage_type,
      region,
      part: null, // not predicted by the baseline
      severity: 'unknown', // not predicted by the baseline
      review_status: 'pending_review',
      notes: '',
      quote_inr: null,
      same_damage_as: null,
      created_at: timestamp,
      updated_at: timestamp,
    };
  });
}

export function manualFinding(photoId: string, region: Region): Finding {
  const timestamp = now();
  return {
    id: newId('finding'),
    photo_id: photoId,
    origin: 'human_annotation',
    original: null,
    damage_type: 'other_uncertain',
    region,
    part: null,
    severity: 'unknown',
    review_status: 'pending_review',
    notes: '',
    quote_inr: null,
    same_damage_as: null,
    created_at: timestamp,
    updated_at: timestamp,
  };
}

export type FindingPatch = Partial<
  Pick<Finding, 'damage_type' | 'part' | 'severity' | 'review_status' | 'notes' | 'quote_inr' | 'same_damage_as'>
>;

/** Apply reviewer edits. The model's original prediction is never modified. */
export function applyPatch(finding: Finding, patch: FindingPatch): Finding {
  const next: Finding = { ...finding, ...patch, original: finding.original, updated_at: now() };
  if (next.same_damage_as === next.id) next.same_damage_as = null;
  if (next.quote_inr !== null && (!Number.isFinite(next.quote_inr) || next.quote_inr < 0)) next.quote_inr = null;
  return next;
}

/** A model output whose label or region a reviewer has changed. */
export function isEditedPrediction(finding: Finding): boolean {
  if (finding.origin !== 'model_output' || !finding.original) return false;
  return finding.damage_type !== finding.original.damage_type || !regionsEqual(finding.region, finding.original.region);
}

export function originLabel(finding: Finding): string {
  switch (finding.origin) {
    case 'illustrative_sample':
      return 'Illustrative sample';
    case 'human_annotation':
      return 'Human annotation';
    case 'model_output':
      return isEditedPrediction(finding) ? 'Model output · edited by reviewer' : 'Model output';
  }
}

export interface QuoteEntry {
  finding_id: string;
  photo_id: string;
  damage_type: string;
  review_status: string;
  quote_inr: number;
  counted: boolean;
  reason: string;
}

export interface QuoteSummary {
  currency: 'INR';
  entries: QuoteEntry[];
  provisional_subtotal_inr: number;
  label: string;
}

export const QUOTE_LABEL =
  'Provisional, partial sum of workshop quotes entered by the reviewer. Not a repair estimate, ' +
  'claim value or final bill. Counts confirmed findings only; findings linked as the same damage ' +
  'as another finding are not counted again. Duplicates across photos that were not linked are ' +
  'still double-counted: reviewer de-duplication is required before any real use.';

export function quoteSummary(findings: Finding[]): QuoteSummary {
  const entries: QuoteEntry[] = [];
  let subtotal = 0;
  for (const finding of findings) {
    if (finding.quote_inr === null) continue;
    let counted = false;
    let reason: string;
    if (finding.review_status !== 'confirmed') reason = 'not confirmed by reviewer';
    else if (finding.same_damage_as) reason = 'linked as the same damage as another finding';
    else {
      counted = true;
      reason = 'counted';
      subtotal += finding.quote_inr;
    }
    entries.push({
      finding_id: finding.id,
      photo_id: finding.photo_id,
      damage_type: finding.damage_type,
      review_status: finding.review_status,
      quote_inr: finding.quote_inr,
      counted,
      reason,
    });
  }
  return { currency: 'INR', entries, provisional_subtotal_inr: Math.round(subtotal * 100) / 100, label: QUOTE_LABEL };
}

export function formatInr(value: number): string {
  return `INR ${value.toLocaleString('en-IN', { minimumFractionDigits: 0, maximumFractionDigits: 2 })}`;
}

/** Display numbers: 1-based position of each finding within its photo. */
export function findingNumbers(findings: Finding[]): Map<string, number> {
  const perPhoto = new Map<string, number>();
  const numbers = new Map<string, number>();
  for (const finding of findings) {
    const next = (perPhoto.get(finding.photo_id) ?? 0) + 1;
    perPhoto.set(finding.photo_id, next);
    numbers.set(finding.id, next);
  }
  return numbers;
}
