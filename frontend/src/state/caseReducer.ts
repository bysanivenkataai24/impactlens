import { applyPatch, type FindingPatch } from '../lib/findings';
import { newId } from '../lib/photos';
import type { Finding, InspectionCase, ModelRun, Photo, VehicleContext } from '../lib/types';

export function emptyCase(): InspectionCase {
  const timestamp = new Date().toISOString();
  return {
    id: newId('case'),
    mode: 'live',
    title: '',
    created_at: timestamp,
    updated_at: timestamp,
    vehicle: { make: '', model: '', year: '', body_style: '', colour: '', notes: '' },
    reviewer_name: '',
    reviewer_notes: '',
    photos: [],
    findings: [],
  };
}

export type CaseAction =
  | { type: 'replace_case'; inspection: InspectionCase }
  | { type: 'set_details'; patch: Partial<Pick<InspectionCase, 'title' | 'reviewer_name' | 'reviewer_notes'>> }
  | { type: 'set_vehicle'; patch: Partial<VehicleContext> }
  | { type: 'add_photos'; photos: Photo[] }
  | { type: 'remove_photo'; photoId: string }
  | { type: 'attach_photo'; photoId: string; objectUrl: string }
  | { type: 'model_result'; photoId: string; run: ModelRun; findings: Finding[] }
  | { type: 'add_finding'; finding: Finding }
  | { type: 'update_finding'; findingId: string; patch: FindingPatch }
  | { type: 'delete_finding'; findingId: string };

const touch = (inspection: InspectionCase): InspectionCase => ({ ...inspection, updated_at: new Date().toISOString() });

export function caseReducer(state: InspectionCase, action: CaseAction): InspectionCase {
  switch (action.type) {
    case 'replace_case':
      return action.inspection;
    case 'set_details':
      return touch({ ...state, ...action.patch });
    case 'set_vehicle':
      return touch({ ...state, vehicle: { ...state.vehicle, ...action.patch } });
    case 'add_photos':
      return touch({ ...state, photos: [...state.photos, ...action.photos] });
    case 'remove_photo': {
      const removed = new Set(state.findings.filter((f) => f.photo_id === action.photoId).map((f) => f.id));
      return touch({
        ...state,
        photos: state.photos.filter((p) => p.id !== action.photoId),
        findings: state.findings
          .filter((f) => f.photo_id !== action.photoId)
          .map((f) => (f.same_damage_as && removed.has(f.same_damage_as) ? { ...f, same_damage_as: null } : f)),
      });
    }
    case 'attach_photo':
      return { ...state, photos: state.photos.map((p) => (p.id === action.photoId ? { ...p, object_url: action.objectUrl } : p)) };
    case 'model_result': {
      // A new run replaces this photo's previous model output; human annotations stay.
      const replaced = new Set(
        state.findings.filter((f) => f.photo_id === action.photoId && f.origin === 'model_output').map((f) => f.id),
      );
      return touch({
        ...state,
        photos: state.photos.map((p) => (p.id === action.photoId ? { ...p, model_run: action.run } : p)),
        findings: [
          ...state.findings
            .filter((f) => !replaced.has(f.id))
            .map((f) => (f.same_damage_as && replaced.has(f.same_damage_as) ? { ...f, same_damage_as: null } : f)),
          ...action.findings,
        ],
      });
    }
    case 'add_finding':
      return touch({ ...state, findings: [...state.findings, action.finding] });
    case 'update_finding':
      return touch({
        ...state,
        findings: state.findings.map((f) => (f.id === action.findingId ? applyPatch(f, action.patch) : f)),
      });
    case 'delete_finding': {
      const target = state.findings.find((f) => f.id === action.findingId);
      // Model output and samples are kept for the audit trail: reject them instead.
      if (!target || target.origin !== 'human_annotation') return state;
      return touch({
        ...state,
        findings: state.findings
          .filter((f) => f.id !== action.findingId)
          .map((f) => (f.same_damage_as === action.findingId ? { ...f, same_damage_as: null } : f)),
      });
    }
  }
}
