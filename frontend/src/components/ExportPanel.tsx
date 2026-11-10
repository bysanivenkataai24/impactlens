import { useState } from 'react';

import { renderAnnotatedImage, photoSource } from '../lib/annotate';
import { downloadBlob } from '../lib/download';
import { formatInr, quoteSummary } from '../lib/findings';
import type { PdfImage } from '../lib/pdf';
import { buildReport, reportFileStem } from '../lib/report';
import { saveCase, storageAvailable } from '../lib/storage';
import { DAMAGE_LABELS, type InspectionCase, type Photo } from '../lib/types';

interface Props {
  inspection: InspectionCase;
  currentPhoto: Photo | null;
  announce: (message: string) => void;
}

export function ExportPanel({ inspection, currentPhoto, announce }: Props) {
  const [busy, setBusy] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const quotes = quoteSummary(inspection.findings);
  const stem = reportFileStem(inspection);
  const empty = inspection.photos.length === 0;

  async function run(label: string, task: () => Promise<void>) {
    setBusy(label);
    setProblem(null);
    try {
      await task();
    } catch (error) {
      setProblem(error instanceof Error ? error.message : 'Export failed.');
    } finally {
      setBusy(null);
    }
  }

  const exportJson = () =>
    run('json', async () => {
      const report = buildReport(inspection);
      downloadBlob(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }), `${stem}.json`);
      announce('JSON report downloaded.');
    });

  const exportImage = () =>
    run('image', async () => {
      if (!currentPhoto) return;
      const blob = await renderAnnotatedImage(inspection, currentPhoto);
      const position = inspection.photos.findIndex((p) => p.id === currentPhoto.id) + 1;
      downloadBlob(blob, `${stem}-photo${position}-annotated.jpg`);
      announce('Annotated image downloaded.');
    });

  const exportPdf = () =>
    run('pdf', async () => {
      const report = buildReport(inspection);
      const images: PdfImage[] = [];
      for (const photo of inspection.photos) {
        if (!photoSource(photo)) continue;
        const blob = await renderAnnotatedImage(inspection, photo, { quality: 0.85 });
        images.push({ photoId: photo.id, jpeg: new Uint8Array(await blob.arrayBuffer()), aspect: photo.width / (photo.height * 1.07) });
      }
      // pdf-lib is loaded only when a PDF is requested.
      const { buildPdf } = await import('../lib/pdf');
      const bytes = await buildPdf(report, images);
      downloadBlob(new Blob([bytes as Uint8Array<ArrayBuffer>], { type: 'application/pdf' }), `${stem}.pdf`);
      announce('PDF assessment downloaded.');
    });

  const saveLocally = () => {
    if (saveCase(inspection)) announce('Case saved in this browser (without photo bytes).');
    else setProblem('This browser refused local storage (private mode, disabled or full). Use the JSON export instead.');
  };

  return (
    <section className="panel" aria-labelledby="export-heading">
      <h2 id="export-heading">Export and save</h2>
      {inspection.mode === 'illustrative_demo' && <p className="notice notice-warn">Exports from the demo are marked ILLUSTRATIVE DEMO.</p>}
      <div className="button-row">
        <button type="button" className="btn" onClick={exportJson} disabled={empty || busy !== null}>
          {busy === 'json' ? 'Preparing…' : 'JSON report'}
        </button>
        <button type="button" className="btn" onClick={exportImage} disabled={!currentPhoto || !photoSource(currentPhoto) || busy !== null}>
          {busy === 'image' ? 'Rendering…' : 'Annotated image (this photo)'}
        </button>
        <button type="button" className="btn" onClick={exportPdf} disabled={empty || busy !== null}>
          {busy === 'pdf' ? 'Building PDF…' : 'PDF assessment'}
        </button>
        <button type="button" className="btn btn-secondary" onClick={saveLocally} disabled={empty || !storageAvailable()}>
          Save in this browser
        </button>
      </div>
      <p className="help">
        Reports carry source and model metadata and each finding's review status. Photos are identified by SHA-256; their bytes are not
        included in the JSON report. Saving in this browser keeps metadata and findings only (no photo bytes) and is not a shared or
        durable database.
      </p>
      {problem && (
        <p className="notice notice-error" role="alert">
          {problem}
        </p>
      )}

      <h3>Reviewer-entered workshop quotes</h3>
      {quotes.entries.length === 0 ? (
        <p className="empty">No quotes entered. Quotes are optional and entered per finding by the reviewer.</p>
      ) : (
        <div className="table-scroll">
          <table className="quotes" data-testid="quote-table">
            <caption className="sr-only">Workshop quotes entered by the reviewer</caption>
            <thead>
              <tr>
                <th scope="col">Photo</th>
                <th scope="col">Finding</th>
                <th scope="col">Quote</th>
                <th scope="col">In subtotal</th>
              </tr>
            </thead>
            <tbody>
              {quotes.entries.map((entry) => (
                <tr key={entry.finding_id}>
                  <td>{inspection.photos.findIndex((p) => p.id === entry.photo_id) + 1}</td>
                  <td>{DAMAGE_LABELS[entry.damage_type as keyof typeof DAMAGE_LABELS] ?? entry.damage_type}</td>
                  <td>{formatInr(entry.quote_inr)}</td>
                  <td>{entry.counted ? 'Yes' : `No: ${entry.reason}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="subtotal" data-testid="quote-subtotal">
        Provisional partial subtotal: <strong>{formatInr(quotes.provisional_subtotal_inr)}</strong>
      </p>
      <p className="help warn-text">{quotes.label}</p>
    </section>
  );
}
