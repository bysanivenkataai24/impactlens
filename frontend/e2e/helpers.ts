import { expect, type Page } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const KEY = 'test-key-0123456789abcdefghij';
export const FIXTURES = path.join(path.dirname(fileURLToPath(import.meta.url)), '.fixtures');
export const fixture = (name: string) => path.join(FIXTURES, name);

/** Collect CSP violations and script errors so a test can assert there were none. */
export function watchConsole(page: Page): string[] {
  const problems: string[] = [];
  page.on('console', (message) => {
    const text = message.text();
    if (message.type() === 'error' || /Content Security Policy|Refused to/i.test(text)) problems.push(text);
  });
  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message}`));
  return problems;
}

export async function connect(page: Page, key = KEY) {
  await page.getByRole('button', { name: 'Model connection' }).click();
  await page.getByLabel('Service key').fill(key);
  await page.getByRole('button', { name: 'Check connection' }).click();
  await expect(page.getByTestId('readiness')).toBeVisible();
  await page.getByRole('button', { name: 'Inspect' }).click();
}

export async function addPhotos(page: Page, ...names: string[]) {
  await page.locator('#photo-input').setInputFiles(names.map(fixture));
}

/**
 * Where a finding's polygon is drawn on screen, normalized to the displayed
 * image. It projects the polygon's geometry through the SVG's on-screen box,
 * so it checks both the coordinates and that the overlay covers the image.
 */
export async function regionOnScreen(page: Page, number: number): Promise<number[]> {
  return page.evaluate((n) => {
    const polygon = document.querySelector(`[data-testid="region-${n}"]`) as SVGPolygonElement | null;
    if (!polygon || !polygon.ownerSVGElement) throw new Error(`region ${n} not drawn`);
    const svg = polygon.ownerSVGElement;
    const image = (document.querySelector('.stage-img') as HTMLImageElement).getBoundingClientRect();
    const box = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    const geometry = polygon.getBBox();
    const sx = box.width / viewBox.width;
    const sy = box.height / viewBox.height;
    const left = box.left + geometry.x * sx;
    const top = box.top + geometry.y * sy;
    return [
      (left - image.left) / image.width,
      (top - image.top) / image.height,
      (left + geometry.width * sx - image.left) / image.width,
      (top + geometry.height * sy - image.top) / image.height,
    ];
  }, number);
}

export function expectBox(actual: number[], expected: number[], tolerance = 0.006) {
  actual.forEach((value, i) => expect(Math.abs(value - expected[i]), `coordinate ${i}: ${actual} vs ${expected}`).toBeLessThan(tolerance));
}

export async function readDownload(page: Page, trigger: () => Promise<void>) {
  const [download] = await Promise.all([page.waitForEvent('download'), trigger()]);
  const file = await download.path();
  const fs = await import('node:fs/promises');
  return { name: download.suggestedFilename(), bytes: await fs.readFile(file) };
}
