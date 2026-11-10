// Browser-local case records for a developer demo.
//
// Stored: case metadata, vehicle context, findings, model metadata and each
// photo's name, size, dimensions and SHA-256. NOT stored: photo bytes, object
// URLs, the service key. This is one browser's storage, not a team database:
// clearing site data, a private window or another device means it is gone.

import type { InspectionCase } from './types';

const KEY = 'impactlens.cases.v1';
const MAX_CASES = 50;

export interface StoredCaseSummary {
  id: string;
  title: string;
  mode: InspectionCase['mode'];
  updated_at: string;
  photos: number;
  findings: number;
}

function readAll(): InspectionCase[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as InspectionCase[]).filter((c) => c && typeof c.id === 'string') : [];
  } catch {
    return [];
  }
}

function writeAll(cases: InspectionCase[]): boolean {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(cases));
    return true;
  } catch {
    return false; // storage disabled, full, or private mode
  }
}

/** Strip everything that must not persist. */
export function toStorable(inspection: InspectionCase): InspectionCase {
  return {
    ...inspection,
    photos: inspection.photos.map((photo) => ({ ...photo, object_url: null })),
  };
}

export function listCases(): StoredCaseSummary[] {
  return readAll()
    .map((c) => ({ id: c.id, title: c.title, mode: c.mode, updated_at: c.updated_at, photos: c.photos.length, findings: c.findings.length }))
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
}

export function saveCase(inspection: InspectionCase): boolean {
  const others = readAll().filter((c) => c.id !== inspection.id);
  return writeAll([toStorable(inspection), ...others].slice(0, MAX_CASES));
}

export function loadCase(id: string): InspectionCase | null {
  return readAll().find((c) => c.id === id) ?? null;
}

export function deleteCase(id: string): boolean {
  return writeAll(readAll().filter((c) => c.id !== id));
}

export function deleteAllCases(): boolean {
  try {
    window.localStorage.removeItem(KEY);
    return true;
  } catch {
    return false;
  }
}

export function storageAvailable(): boolean {
  try {
    const probe = `${KEY}.probe`;
    window.localStorage.setItem(probe, '1');
    window.localStorage.removeItem(probe);
    return true;
  } catch {
    return false;
  }
}

const URL_KEY = 'impactlens.service_url';

/** The service URL is not a secret; remembering it is a convenience. The key is never stored. */
export function rememberServiceUrl(url: string): void {
  try {
    if (url) window.localStorage.setItem(URL_KEY, url);
    else window.localStorage.removeItem(URL_KEY);
  } catch {
    /* optional convenience */
  }
}

export function recallServiceUrl(): string {
  try {
    return window.localStorage.getItem(URL_KEY) ?? '';
  } catch {
    return '';
  }
}
