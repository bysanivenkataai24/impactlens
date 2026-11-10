// End-to-end tests against the real service with a SYNTHETIC adapter (not a model).

import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

import { addPhotos, connect, expectBox, fixture, KEY, readDownload, regionOnScreen, watchConsole } from './helpers';

test.describe('upload, inference and overlay alignment', () => {
  test('portrait (EXIF 6) and landscape photos keep masks aligned', async ({ page }) => {
    const problems = watchConsole(page);
    await page.goto('/');
    await connect(page);
    await expect(page.getByTestId('conn-pill')).toContainText('Model ready');

    await addPhotos(page, 'portrait-exif6.jpg', 'landscape.png', 'not-a-photo.jpg');
    await expect(page.getByTestId('rejected-files')).toContainText('not-a-photo.jpg');
    await expect(page.getByTestId('rejected-files')).toContainText('Not a JPEG, PNG or WebP');
    await expect(page.getByTestId('thumb-2')).toBeVisible();

    // Nothing is sent until the user asks.
    await page.getByTestId('thumb-1').click();
    await expect(page.getByTestId('region-1')).toHaveCount(0);

    await page.getByTestId('run-model').click();
    await expect(page.getByTestId('region-1')).toBeVisible();
    await expect(page.getByTestId('model-outcome')).toContainText('1 region(s) returned for human review');

    // The browser shows the photo upright (600x900), like the service.
    const natural = await page.locator('.stage-img').evaluate((img: HTMLImageElement) => [img.naturalWidth, img.naturalHeight]);
    expect(natural).toEqual([600, 900]);
    expectBox(await regionOnScreen(page, 1), [0.1, 0.1, 0.4, 0.3]);

    // Switching photos keeps each photo's findings.
    await page.getByTestId('thumb-2').click();
    await expect(page.getByTestId('region-1')).toHaveCount(0);
    await page.getByTestId('run-model').click();
    await expect(page.getByTestId('region-1')).toBeVisible();
    expectBox(await regionOnScreen(page, 1), [0.5, 0.5, 0.75, 0.75]);
    await page.getByTestId('thumb-1').click();
    await expect(page.getByTestId('region-1')).toBeVisible();
    expectBox(await regionOnScreen(page, 1), [0.1, 0.1, 0.4, 0.3]);

    // Alignment survives a layout change.
    await page.setViewportSize({ width: 700, height: 1000 });
    expectBox(await regionOnScreen(page, 1), [0.1, 0.1, 0.4, 0.3]);

    // Overlay toggle and highlight from the findings list.
    await page.getByLabel('Overlays').uncheck();
    await expect(page.getByTestId('region-1')).toHaveCount(0);
    await page.getByLabel('Overlays').check();
    await page.getByTestId('finding-1').click();
    await expect(page.locator('.region.is-selected')).toHaveCount(1);

    expect(problems).toEqual([]);
  });

  test('lossless WebP is accepted and aligned', async ({ page }) => {
    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'square.webp');
    await page.getByTestId('run-model').click();
    await expect(page.getByTestId('region-1')).toBeVisible();
    expectBox(await regionOnScreen(page, 1), [0.25, 0.25, 0.5, 0.5]);
  });

  test('no detection is not reported as undamaged', async ({ page }) => {
    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'no-region.png');
    await page.getByTestId('run-model').click();
    const outcome = page.getByTestId('model-outcome');
    await expect(outcome).toContainText('No supported damage region was returned');
    await expect(outcome).toContainText('does not mean the vehicle is undamaged');
    await expect(page.getByText('An empty list is not evidence that the vehicle is undamaged')).toBeVisible();
  });

  test('a wrong service key is reported as an authentication problem', async ({ page }) => {
    await page.goto('/');
    await connect(page, 'wrong-key-wrong-key-wrong-key');
    await addPhotos(page, 'landscape.png');
    await page.getByTestId('run-model').click();
    await expect(page.getByTestId('inference-status').locator('[data-error-kind="unauthenticated"]')).toBeVisible();
    await expect(page.getByTestId('region-1')).toHaveCount(0);
  });
});

test.describe('service failure states (responses intercepted in the browser)', () => {
  const envelope = (code: string, message: string, extra: Record<string, unknown> = {}) =>
    JSON.stringify({ schema_version: '1.0.0', error: { code, message, request_id: 'r1', ...extra } });

  const cases: Array<[string, number, string, string, string]> = [
    ['model unavailable', 503, 'application/json', envelope('model_unavailable', 'No model', { reasons: ['class_mapping_mismatch'] }), 'model_unavailable'],
    ['cold start proxy page', 503, 'text/html', '<html><body>Your Space is starting</body></html>', 'cold_start'],
    ['server timeout', 504, 'application/json', envelope('inference_timeout', 'Out of time', { retry_after_s: 15 }), 'server_timeout'],
    ['busy', 429, 'application/json', envelope('server_busy', 'At capacity', { retry_after_s: 5 }), 'busy'],
    ['rejected photo', 413, 'application/json', envelope('image_pixel_limit_exceeded', 'Too many pixels'), 'rejected_input'],
    ['invalid response', 200, 'application/json', JSON.stringify({ schema_version: '1.0.0', detections: [{ bbox: [0, 0, 9, 9] }] }), 'invalid_response'],
  ];

  for (const [name, status, contentType, body, kind] of cases) {
    test(name, async ({ page }) => {
      await page.goto('/');
      await connect(page);
      await addPhotos(page, 'landscape.png');
      await page.route('**/predict', (route) => route.fulfill({ status, contentType, body }));
      await page.getByTestId('run-model').click();
      await expect(page.getByTestId('inference-status').locator(`[data-error-kind="${kind}"]`)).toBeVisible();
      await expect(page.getByTestId('region-1')).toHaveCount(0);
      await expect(page.getByTestId('model-outcome')).toHaveCount(0);
    });
  }

  test('a response for different bytes is refused', async ({ page, request }) => {
    // A genuine service response for this photo, then altered as if it described other bytes.
    const fs = await import('node:fs/promises');
    const genuine = await request.post('/predict', {
      headers: { Authorization: `Bearer ${KEY}` },
      multipart: { image: { name: 'landscape.png', mimeType: 'image/png', buffer: await fs.readFile(fixture('landscape.png')) } },
    });
    expect(genuine.status()).toBe(200);
    const json = await genuine.json();
    json.image.sha256 = 'f'.repeat(64);

    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'landscape.png');
    await page.route('**/predict', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(json) }));
    await page.getByTestId('run-model').click();
    await expect(page.getByText('Response does not match this photo')).toBeVisible();
    await expect(page.getByTestId('region-1')).toHaveCount(0);
  });

  test('cancel while waiting', async ({ page }) => {
    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'landscape.png');
    await page.route('**/predict', () => {
      /* never answer */
    });
    await page.getByTestId('run-model').click();
    await expect(page.getByText(/Waiting for the model/)).toBeVisible();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByTestId('inference-status').locator('[data-error-kind="cancelled"]')).toBeVisible();
  });
});

test.describe('human review and export', () => {
  test('draw a region, edit findings, export JSON, image and PDF', async ({ page }) => {
    const problems = watchConsole(page);
    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'landscape.png');
    await page.getByTestId('run-model').click();
    await expect(page.getByTestId('region-1')).toBeVisible();

    // Relabel the model output: the original prediction is preserved and marked as edited.
    await page.getByTestId('finding-1').click();
    await page.getByLabel('Damage type').selectOption('scratch');
    await expect(page.getByTestId('edited-note')).toContainText('changed the label from Dent to Scratch');
    await expect(page.getByTestId('finding-1')).toContainText('Model output · edited by reviewer');
    await page.getByLabel('Affected part').fill('Rear left door');
    await page.getByLabel('Visible severity').selectOption('minor');
    await page.getByLabel('Workshop quote (INR)').fill('4500');
    await page.getByRole('button', { name: 'Confirm' }).click();
    await expect(page.getByTestId('finding-1')).toContainText('Confirmed by reviewer');
    await expect(page.getByTestId('quote-subtotal')).toContainText('INR 4,500');

    // Draw a manual polygon on the photo.
    await page.getByRole('button', { name: 'Draw region' }).click();
    const box = (await page.getByTestId('overlay').boundingBox())!;
    for (const [x, y] of [
      [0.1, 0.1],
      [0.3, 0.1],
      [0.3, 0.3],
      [0.1, 0.3],
    ]) {
      await page.mouse.click(box.x + box.width * x, box.y + box.height * y);
    }
    await page.getByRole('button', { name: /Finish region/ }).click();
    await expect(page.getByTestId('finding-2')).toContainText('Human annotation');
    expectBox(await regionOnScreen(page, 2), [0.1, 0.1, 0.3, 0.3], 0.01);
    await page.getByLabel('Damage type').selectOption('other_uncertain');
    await page.getByLabel('Notes on this finding').fill('Paint transfer or scuff; needs closer photo.');
    await page.getByLabel('Review status').selectOption('needs_more_evidence');

    // Keyboard alternative: rectangle by numbers.
    await page.getByText('Add a rectangular region by numbers').click();
    await page.getByLabel('Left edge (%)').fill('60');
    await page.getByLabel('Top edge (%)').fill('10');
    await page.getByLabel('Width (%)').fill('20');
    await page.getByLabel('Height (%)').fill('15');
    await page.getByRole('button', { name: 'Add region' }).click();
    await expect(page.getByTestId('finding-3')).toContainText('Human annotation');

    await page.getByLabel('Make').fill('Maruti Suzuki');
    await page.getByLabel('Model', { exact: true }).fill('Swift');
    await page.getByLabel('Year').fill('2021');

    const json = await readDownload(page, () => page.getByRole('button', { name: 'JSON report' }).click());
    expect(json.name).toMatch(/\.json$/);
    const report = JSON.parse(json.bytes.toString('utf8'));
    expect(report.report_type).toBe('impactlens.assessment');
    expect(report.report_version).toBe('1.0.0');
    expect(report.vehicle).toMatchObject({ make: 'Maruti Suzuki', model: 'Swift', year: '2021' });
    const findings = report.photos[0].findings;
    expect(findings).toHaveLength(3);
    expect(findings[0]).toMatchObject({
      origin: 'model_output',
      edited_by_reviewer: true,
      damage_type: 'scratch',
      part: 'Rear left door',
      part_source: 'reviewer',
      visible_severity: 'minor',
      severity_source: 'reviewer',
      review_status: 'confirmed',
      quote_inr: 4500,
    });
    expect(findings[0].original_prediction).toMatchObject({ damage_type: 'dent', model_id: 'synthetic-test-double' });
    expect(findings[1]).toMatchObject({ origin: 'human_annotation', damage_type: 'other_uncertain', review_status: 'needs_more_evidence' });
    expect(report.photos[0].photo_bytes_included).toBe(false);
    expect(report.photos[0].model_run.checkpoint_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(json.bytes.toString('utf8')).not.toContain('blob:');
    expect(report.quotes.provisional_subtotal_inr).toBe(4500);

    const image = await readDownload(page, () => page.getByRole('button', { name: 'Annotated image (this photo)' }).click());
    expect(image.name).toMatch(/annotated\.jpg$/);
    expect([...image.bytes.subarray(0, 3)]).toEqual([0xff, 0xd8, 0xff]);
    expect(image.bytes.includes(Buffer.from('Exif'))).toBe(false);

    const pdf = await readDownload(page, () => page.getByRole('button', { name: 'PDF assessment' }).click());
    expect(pdf.name).toMatch(/\.pdf$/);
    expect(pdf.bytes.subarray(0, 5).toString()).toBe('%PDF-');
    expect(pdf.bytes.length).toBeGreaterThan(10_000);

    expect(problems).toEqual([]);
  });

  test('saved cases exclude photo bytes, need re-attachment and can be deleted', async ({ page }) => {
    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'landscape.png');
    await page.getByTestId('run-model').click();
    await expect(page.getByTestId('region-1')).toBeVisible();
    await page.getByLabel('Case title').fill('Saved intake');
    await page.getByRole('button', { name: 'Save in this browser' }).click();
    const stored = await page.evaluate(() => window.localStorage.getItem('impactlens.cases.v1') ?? '');
    expect(stored).toContain('Saved intake');
    expect(stored).not.toContain('blob:');
    expect(stored).not.toContain(KEY);

    await page.reload();
    await page.getByRole('button', { name: 'Saved cases' }).click();
    await expect(page.getByText('Photo bytes are never saved')).toBeVisible();
    await page.getByRole('button', { name: 'Load' }).click();
    await expect(page.getByText('Photo not stored.')).toBeVisible();

    await page.getByLabel(/Re-attach/).setInputFiles(fixture('no-region.png'));
    await expect(page.getByTestId('rejected-files')).toContainText('SHA-256 differs');
    await page.getByLabel(/Re-attach/).setInputFiles(fixture('landscape.png'));
    await expect(page.getByTestId('region-1')).toBeVisible();

    await page.getByRole('button', { name: 'Saved cases' }).click();
    await page.getByRole('button', { name: 'Delete', exact: true }).click();
    await expect(page.getByText('No saved cases.')).toBeVisible();
  });
});

test.describe('illustrative demo', () => {
  test('works without a model and is never sent to one', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Open illustrative demo' }).click();
    await expect(page.getByTestId('demo-banner')).toContainText('not model output');
    await expect(page.getByTestId('region-1')).toBeVisible();
    await expect(page.getByTestId('finding-1')).toContainText('Illustrative sample');
    await expect(page.getByTestId('run-model')).toBeDisabled();
    await expect(page.getByText('Demo photos are illustrations and are never sent to a model.')).toBeVisible();
    const json = await readDownload(page, () => page.getByRole('button', { name: 'JSON report' }).click());
    expect(json.name.startsWith('DEMO-')).toBe(true);
    expect(JSON.parse(json.bytes.toString('utf8')).notice).toContain('ILLUSTRATIVE DEMO');
  });
});

test.describe('accessibility and layout', () => {
  test('no serious accessibility violations on any view', async ({ page }) => {
    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'landscape.png');
    await page.getByTestId('run-model').click();
    await page.getByTestId('finding-1').click();
    for (const view of ['Inspect', 'Model connection', 'Saved cases', 'About & limits']) {
      await page.getByRole('button', { name: view, exact: true }).click();
      const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa']).analyze();
      const serious = results.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
      expect(serious.map((v) => `${view}: ${v.id} (${v.nodes.length})`)).toEqual([]);
    }
  });

  test('keyboard: skip link and findings reachable without a pointer', async ({ page }) => {
    await page.goto('/');
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Skip to main content' })).toBeFocused();
    await page.getByRole('button', { name: 'Open illustrative demo' }).click();
    const first = page.getByTestId('finding-1');
    await first.focus();
    await page.keyboard.press('Enter');
    await expect(first).toHaveAttribute('aria-expanded', 'true');
    await expect(page.locator('.region.is-selected')).toHaveCount(1);
  });

  test('no horizontal page scroll', async ({ page }) => {
    await page.goto('/');
    await connect(page);
    await addPhotos(page, 'portrait-exif6.jpg', 'landscape.png');
    await page.getByTestId('run-model').click();
    await page.getByTestId('finding-1').click();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });
});
