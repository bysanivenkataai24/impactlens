import { describe, expect, it } from 'vitest';

import { applyPatch, findingsFromResponse, isEditedPrediction, manualFinding, modelRunFrom, originLabel, quoteSummary } from '../src/lib/findings';
import { regionFromPercentRect, regionFromPolygon } from '../src/lib/geometry';
import { sniffMediaType } from '../src/lib/photos';
import { buildReport } from '../src/lib/report';
import type { Photo } from '../src/lib/types';
import { caseReducer, emptyCase } from '../src/state/caseReducer';
import { predictResponse } from './fixtures';

const photo = (id: string): Photo => ({
  id,
  file_name: `${id}.jpg`,
  media_type: 'image/jpeg',
  bytes: 10,
  sha256: id.padEnd(64, '0'),
  width: 400,
  height: 200,
  object_url: `blob:${id}`,
  model_run: null,
});

describe('findings', () => {
  it('keeps the original prediction when a reviewer relabels it', () => {
    const [finding] = findingsFromResponse('p1', predictResponse());
    expect(originLabel(finding)).toBe('Model output');
    const edited = applyPatch(finding, { damage_type: 'scratch', part: 'Rear left door', severity: 'minor' });
    expect(edited.original?.damage_type).toBe('dent');
    expect(edited.original?.score).toBe(0.82);
    expect(isEditedPrediction(edited)).toBe(true);
    expect(originLabel(edited)).toBe('Model output · edited by reviewer');
    expect(() => {
      (edited.original as { damage_type: string }).damage_type = 'crack';
    }).toThrow();
  });

  it('does not treat reviewer notes or status as an edit of the prediction', () => {
    const [finding] = findingsFromResponse('p1', predictResponse());
    expect(isEditedPrediction(applyPatch(finding, { review_status: 'confirmed', notes: 'ok', part: 'Bonnet' }))).toBe(false);
  });

  it('labels manual regions as human annotations with unknown severity and no part', () => {
    const region = regionFromPercentRect(10, 10, 20, 20)!;
    const finding = manualFinding('p1', region);
    expect(originLabel(finding)).toBe('Human annotation');
    expect(finding.damage_type).toBe('other_uncertain');
    expect(finding.part).toBeNull();
    expect(finding.severity).toBe('unknown');
  });

  it('rejects degenerate regions', () => {
    expect(regionFromPolygon([[0.1, 0.1], [0.2, 0.2]])).toBeNull();
    expect(regionFromPolygon([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])).toBeNull();
    expect(regionFromPercentRect(10, 10, 0, 5)).toBeNull();
    expect(regionFromPercentRect(90, 90, 50, 50)?.bbox).toEqual([0.9, 0.9, 1, 1]);
  });

  it('only sums confirmed, unlinked quotes and says it is provisional', () => {
    const [a] = findingsFromResponse('p1', predictResponse());
    const [b] = findingsFromResponse('p2', predictResponse());
    const c = manualFinding('p1', regionFromPercentRect(1, 1, 5, 5)!);
    const findings = [
      applyPatch(a, { review_status: 'confirmed', quote_inr: 4500 }),
      applyPatch(b, { review_status: 'confirmed', quote_inr: 4500, same_damage_as: a.id }),
      applyPatch(c, { review_status: 'pending_review', quote_inr: 900 }),
    ];
    const summary = quoteSummary(findings);
    expect(summary.provisional_subtotal_inr).toBe(4500);
    expect(summary.entries.map((e) => e.counted)).toEqual([true, false, false]);
    expect(summary.label).toMatch(/Provisional, partial/);
    expect(summary.label).toMatch(/de-duplication/);
  });

  it('drops invalid quotes and self links', () => {
    const [a] = findingsFromResponse('p1', predictResponse());
    expect(applyPatch(a, { quote_inr: -5 }).quote_inr).toBeNull();
    expect(applyPatch(a, { same_damage_as: a.id }).same_damage_as).toBeNull();
  });
});

describe('caseReducer', () => {
  it('a new model run replaces only that photo’s model findings', () => {
    let state = caseReducer(emptyCase(), { type: 'add_photos', photos: [photo('p1'), photo('p2')] });
    const response = predictResponse();
    state = caseReducer(state, { type: 'model_result', photoId: 'p1', run: modelRunFrom(response), findings: findingsFromResponse('p1', response) });
    state = caseReducer(state, { type: 'model_result', photoId: 'p2', run: modelRunFrom(response), findings: findingsFromResponse('p2', response) });
    const manual = manualFinding('p1', regionFromPercentRect(1, 1, 5, 5)!);
    state = caseReducer(state, { type: 'add_finding', finding: manual });
    state = caseReducer(state, { type: 'model_result', photoId: 'p1', run: modelRunFrom(response), findings: findingsFromResponse('p1', response) });
    expect(state.findings.filter((f) => f.photo_id === 'p1')).toHaveLength(2);
    expect(state.findings.filter((f) => f.photo_id === 'p2')).toHaveLength(1);
    expect(state.findings.some((f) => f.id === manual.id)).toBe(true);
  });

  it('model output cannot be deleted, only rejected; annotations can be deleted', () => {
    let state = caseReducer(emptyCase(), { type: 'add_photos', photos: [photo('p1')] });
    const [model] = findingsFromResponse('p1', predictResponse());
    const manual = manualFinding('p1', regionFromPercentRect(1, 1, 5, 5)!);
    state = caseReducer(state, { type: 'add_finding', finding: model });
    state = caseReducer(state, { type: 'add_finding', finding: manual });
    state = caseReducer(state, { type: 'delete_finding', findingId: model.id });
    state = caseReducer(state, { type: 'delete_finding', findingId: manual.id });
    expect(state.findings.map((f) => f.id)).toEqual([model.id]);
  });

  it('removing a photo removes its findings and links to them', () => {
    let state = caseReducer(emptyCase(), { type: 'add_photos', photos: [photo('p1'), photo('p2')] });
    const [a] = findingsFromResponse('p1', predictResponse());
    const [b] = findingsFromResponse('p2', predictResponse());
    state = caseReducer(state, { type: 'add_finding', finding: a });
    state = caseReducer(state, { type: 'add_finding', finding: applyPatch(b, { same_damage_as: a.id }) });
    state = caseReducer(state, { type: 'remove_photo', photoId: 'p1' });
    expect(state.photos.map((p) => p.id)).toEqual(['p2']);
    expect(state.findings).toHaveLength(1);
    expect(state.findings[0].same_damage_as).toBeNull();
  });
});

describe('report', () => {
  it('is versioned, excludes photo bytes and object URLs, and keeps the original prediction', () => {
    let state = caseReducer(emptyCase(), { type: 'add_photos', photos: [photo('p1')] });
    const response = predictResponse();
    state = caseReducer(state, { type: 'model_result', photoId: 'p1', run: modelRunFrom(response), findings: findingsFromResponse('p1', response) });
    state = caseReducer(state, { type: 'update_finding', findingId: state.findings[0].id, patch: { damage_type: 'crack' } });
    const report = buildReport(state, new Date('2026-11-10T00:00:00Z'));
    const text = JSON.stringify(report);
    expect(report.report_type).toBe('impactlens.assessment');
    expect(report.report_version).toBe('1.0.0');
    expect(text).not.toContain('blob:');
    expect(report.photos[0].photo_bytes_included).toBe(false);
    const finding = report.photos[0].findings[0];
    expect(finding.damage_type).toBe('crack');
    expect(finding.original_prediction?.damage_type).toBe('dent');
    expect(finding.edited_by_reviewer).toBe(true);
    expect(finding.severity_source).toBeNull();
    expect(report.review_summary.note).toMatch(/Not a count of unique/);
    expect(report.photos[0].model_run?.checkpoint_sha256).toBe('a'.repeat(64));
  });
});

describe('sniffMediaType', () => {
  it('reads magic bytes, not names', () => {
    expect(sniffMediaType(new Uint8Array([0xff, 0xd8, 0xff, 0xe0]))).toBe('image/jpeg');
    expect(sniffMediaType(new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))).toBe('image/png');
    expect(sniffMediaType(new TextEncoder().encode('RIFF\0\0\0\0WEBPVP8 '))).toBe('image/webp');
    expect(sniffMediaType(new TextEncoder().encode('GIF89a'))).toBeNull();
    expect(sniffMediaType(new TextEncoder().encode('\0\0\0\x18ftypheic'))).toBeNull();
  });
});
