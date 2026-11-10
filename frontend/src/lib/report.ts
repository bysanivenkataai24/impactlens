// Versioned assessment report. The JSON export is the canonical record; the PDF
// and the annotated image are renderings of it.

import { findingNumbers, isEditedPrediction, originLabel, quoteSummary, type QuoteSummary } from './findings';
import type { Finding, InspectionCase, ReviewStatus } from './types';

export const REPORT_TYPE = 'impactlens.assessment';
export const REPORT_VERSION = '1.0.0';
export const APP_VERSION = '0.1.0';

export const REPORT_DISCLAIMERS = [
  'Decision-support record. It is not a claim decision, a roadworthiness or mechanical assessment, a fault determination or a repair estimate.',
  'Model regions are suggestions. Only findings marked "Confirmed by reviewer" were accepted by a person.',
  'Model scores are uncalibrated and are not probabilities.',
  'Affected part, visible severity and workshop quotes are reviewer entries, not model predictions.',
  'Several photos may show the same physical damage. Findings are listed per photo and are not a count of unique damages.',
  'Photo bytes are not included in this report. Each photo is identified by its SHA-256 hash.',
];

export const DEMO_NOTICE =
  'ILLUSTRATIVE DEMO: the photo is a drawing and every finding is a hand-made sample. Nothing here is a model output.';

export interface ReportFinding {
  number: number;
  finding_id: string;
  origin: Finding['origin'];
  origin_label: string;
  edited_by_reviewer: boolean;
  damage_type: string;
  region: Finding['region'];
  part: string | null;
  part_source: 'reviewer' | null;
  visible_severity: Finding['severity'];
  severity_source: 'reviewer' | null;
  review_status: ReviewStatus;
  notes: string;
  quote_inr: number | null;
  same_damage_as: string | null;
  original_prediction: Finding['original'];
  created_at: string;
  updated_at: string;
}

export interface AssessmentReport {
  report_type: typeof REPORT_TYPE;
  report_version: typeof REPORT_VERSION;
  generated_at: string;
  generator: { application: string; version: string };
  mode: InspectionCase['mode'];
  notice: string | null;
  disclaimers: string[];
  case: { id: string; title: string; created_at: string; updated_at: string };
  vehicle: InspectionCase['vehicle'];
  reviewer: { name: string; notes: string };
  photos: Array<{
    photo_id: string;
    position: number;
    file_name: string;
    media_type: string;
    bytes: number;
    sha256: string;
    width: number;
    height: number;
    coordinate_space: string;
    photo_bytes_included: false;
    model_run: InspectionCase['photos'][number]['model_run'];
    findings: ReportFinding[];
  }>;
  review_summary: {
    note: string;
    findings_listed: number;
    by_review_status: Record<ReviewStatus, number>;
    by_origin: Record<Finding['origin'], number>;
    photos_with_model_run: number;
    photos_without_model_run: number;
  };
  quotes: QuoteSummary;
}

export function buildReport(inspection: InspectionCase, generatedAt = new Date()): AssessmentReport {
  const numbers = findingNumbers(inspection.findings);
  const byStatus: Record<ReviewStatus, number> = { pending_review: 0, confirmed: 0, rejected: 0, needs_more_evidence: 0 };
  const byOrigin: Record<Finding['origin'], number> = { illustrative_sample: 0, model_output: 0, human_annotation: 0 };
  for (const finding of inspection.findings) {
    byStatus[finding.review_status] += 1;
    byOrigin[finding.origin] += 1;
  }
  const photosWithRun = inspection.photos.filter((p) => p.model_run).length;
  return {
    report_type: REPORT_TYPE,
    report_version: REPORT_VERSION,
    generated_at: generatedAt.toISOString(),
    generator: { application: 'ImpactLens inspection UI', version: APP_VERSION },
    mode: inspection.mode,
    notice: inspection.mode === 'illustrative_demo' ? DEMO_NOTICE : null,
    disclaimers: REPORT_DISCLAIMERS,
    case: { id: inspection.id, title: inspection.title, created_at: inspection.created_at, updated_at: inspection.updated_at },
    vehicle: { ...inspection.vehicle },
    reviewer: { name: inspection.reviewer_name, notes: inspection.reviewer_notes },
    photos: inspection.photos.map((photo, index) => ({
      photo_id: photo.id,
      position: index + 1,
      file_name: photo.file_name,
      media_type: photo.media_type,
      bytes: photo.bytes,
      sha256: photo.sha256,
      width: photo.width,
      height: photo.height,
      coordinate_space: 'normalized [0,1] to the EXIF-oriented image; origin top-left; bbox [x_min, y_min, x_max, y_max]',
      photo_bytes_included: false,
      model_run: photo.model_run,
      findings: inspection.findings
        .filter((f) => f.photo_id === photo.id)
        .map((f) => ({
          number: numbers.get(f.id) ?? 0,
          finding_id: f.id,
          origin: f.origin,
          origin_label: originLabel(f),
          edited_by_reviewer: isEditedPrediction(f),
          damage_type: f.damage_type,
          region: f.region,
          part: f.part,
          part_source: f.part ? 'reviewer' : null,
          visible_severity: f.severity,
          severity_source: f.severity === 'unknown' ? null : 'reviewer',
          review_status: f.review_status,
          notes: f.notes,
          quote_inr: f.quote_inr,
          same_damage_as: f.same_damage_as,
          original_prediction: f.original,
          created_at: f.created_at,
          updated_at: f.updated_at,
        })),
    })),
    review_summary: {
      note: 'Counts of findings as listed per photo. Not a count of unique physical damages.',
      findings_listed: inspection.findings.length,
      by_review_status: byStatus,
      by_origin: byOrigin,
      photos_with_model_run: photosWithRun,
      photos_without_model_run: inspection.photos.length - photosWithRun,
    },
    quotes: quoteSummary(inspection.findings),
  };
}

export function reportFileStem(inspection: InspectionCase): string {
  const safe = (inspection.title || 'assessment').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40);
  const prefix = inspection.mode === 'illustrative_demo' ? 'DEMO-' : '';
  return `${prefix}impactlens-${safe || 'assessment'}-${inspection.id.slice(-6)}`;
}
