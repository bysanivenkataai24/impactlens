import { PDFDocument } from 'pdf-lib';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { createDemoCase } from '../src/lib/demo';
import { buildPdf, pdfSafe } from '../src/lib/pdf';
import { buildReport } from '../src/lib/report';
import { deleteAllCases, deleteCase, listCases, loadCase, saveCase } from '../src/lib/storage';
import { emptyCase } from '../src/state/caseReducer';

describe('local storage', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
  });

  it('saves without photo bytes or object URLs, and deletes', () => {
    const inspection = { ...emptyCase(), title: 'Intake' };
    inspection.photos = [
      { id: 'p1', file_name: 'a.jpg', media_type: 'image/jpeg', bytes: 10, sha256: 'f'.repeat(64), width: 10, height: 10, object_url: 'blob:secret', model_run: null },
    ];
    expect(saveCase(inspection)).toBe(true);
    const raw = window.localStorage.getItem('impactlens.cases.v1') ?? '';
    expect(raw).not.toContain('blob:');
    expect(loadCase(inspection.id)?.photos[0].object_url).toBeNull();
    expect(listCases()[0]).toMatchObject({ title: 'Intake', photos: 1 });
    deleteCase(inspection.id);
    expect(listCases()).toEqual([]);
    saveCase(inspection);
    deleteAllCases();
    expect(listCases()).toEqual([]);
  });

  it('survives storage being unavailable', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('quota', 'QuotaExceededError');
    });
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('denied', 'SecurityError');
    });
    expect(saveCase(emptyCase())).toBe(false);
    expect(listCases()).toEqual([]);
  });

  it('ignores corrupt stored data', () => {
    window.localStorage.setItem('impactlens.cases.v1', '{not json');
    expect(listCases()).toEqual([]);
  });
});

describe('pdf', () => {
  it('replaces characters the standard fonts cannot encode', () => {
    expect(pdfSafe('Quote ₹4,500')).toBe('Quote INR 4,500');
    expect(pdfSafe('मारुति')).toBe('??????');
    expect(pdfSafe('café – ok')).toBe('café – ok');
  });

  it('builds a document with a summary page plus one page per photo', async () => {
    const report = buildReport(createDemoCase());
    const bytes = await buildPdf(report, []);
    expect(new TextDecoder().decode(bytes.slice(0, 5))).toBe('%PDF-');
    const doc = await PDFDocument.load(bytes);
    expect(doc.getPageCount()).toBeGreaterThanOrEqual(2);
    expect(doc.getTitle()).toMatch(/ImpactLens assessment/);
  });
});

describe('demo mode', () => {
  it('is marked as illustrative throughout', () => {
    const demo = createDemoCase();
    expect(demo.mode).toBe('illustrative_demo');
    expect(demo.findings.every((f) => f.origin === 'illustrative_sample' && f.original === null)).toBe(true);
    const report = buildReport(demo);
    expect(report.notice).toMatch(/ILLUSTRATIVE DEMO/);
    expect(report.review_summary.by_origin.model_output).toBe(0);
  });
});
