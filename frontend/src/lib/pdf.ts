// Printable PDF assessment built with pdf-lib (MIT). Uses the standard
// Helvetica font, which covers Windows-1252 only: other characters are
// replaced with '?'. The JSON report keeps the exact text.

import { PDFDocument, StandardFonts, rgb, type PDFFont, type PDFPage } from 'pdf-lib';

import { formatInr } from './findings';
import type { AssessmentReport } from './report';
import { DAMAGE_LABELS, REVIEW_STATUS_LABELS, SEVERITY_LABELS, type ReviewDamageType } from './types';

const A4: [number, number] = [595.28, 841.89];
const MARGIN = 42;
const INK = rgb(0.07, 0.09, 0.15);
const MUTED = rgb(0.32, 0.36, 0.42);
const WARN = rgb(0.6, 0.18, 0.05);

const CP1252_EXTRA = '€‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ';

/** Make text encodable by the standard PDF fonts. */
export function pdfSafe(text: string): string {
  return Array.from(text.replace(/₹/g, 'INR ').replace(/\s+/g, ' '))
    .map((ch) => {
      const code = ch.codePointAt(0) ?? 63;
      return (code >= 32 && code <= 126) || (code >= 160 && code <= 255) || CP1252_EXTRA.includes(ch) ? ch : '?';
    })
    .join('');
}

class Writer {
  page!: PDFPage;
  y = 0;
  constructor(
    private doc: PDFDocument,
    private font: PDFFont,
    private bold: PDFFont,
    private footer: string,
  ) {
    this.newPage();
  }

  newPage() {
    this.page = this.doc.addPage(A4);
    this.y = A4[1] - MARGIN;
    this.page.drawText(pdfSafe(this.footer), { x: MARGIN, y: 22, size: 7.5, font: this.font, color: MUTED });
  }

  ensure(height: number) {
    if (this.y - height < MARGIN) this.newPage();
  }

  wrap(text: string, size: number, font: PDFFont, width: number): string[] {
    const words = pdfSafe(text).split(' ');
    const lines: string[] = [];
    let line = '';
    for (const word of words) {
      const candidate = line ? `${line} ${word}` : word;
      if (font.widthOfTextAtSize(candidate, size) <= width) line = candidate;
      else {
        if (line) lines.push(line);
        // Hard-break very long tokens (hashes, URLs).
        let rest = word;
        while (font.widthOfTextAtSize(rest, size) > width && rest.length > 1) {
          let cut = rest.length;
          while (cut > 1 && font.widthOfTextAtSize(rest.slice(0, cut), size) > width) cut--;
          lines.push(rest.slice(0, cut));
          rest = rest.slice(cut);
        }
        line = rest;
      }
    }
    if (line) lines.push(line);
    return lines.length ? lines : [''];
  }

  text(text: string, { size = 9.5, bold = false, color = INK, indent = 0, gap = 2 } = {}) {
    const font = bold ? this.bold : this.font;
    for (const line of this.wrap(text, size, font, A4[0] - 2 * MARGIN - indent)) {
      this.ensure(size + gap);
      this.y -= size + gap;
      this.page.drawText(line, { x: MARGIN + indent, y: this.y, size, font, color });
    }
  }

  heading(text: string, size = 13) {
    this.ensure(size + 18);
    this.y -= 10;
    this.text(text, { size, bold: true });
    this.y -= 2;
  }

  rule() {
    this.ensure(8);
    this.y -= 5;
    this.page.drawLine({ start: { x: MARGIN, y: this.y }, end: { x: A4[0] - MARGIN, y: this.y }, thickness: 0.5, color: MUTED });
    this.y -= 3;
  }

  keyValues(rows: Array<[string, string]>) {
    for (const [key, value] of rows) {
      const lines = this.wrap(value || '-', 9.5, this.font, A4[0] - 2 * MARGIN - 130);
      this.ensure(12 * lines.length);
      this.y -= 11.5;
      this.page.drawText(pdfSafe(key), { x: MARGIN, y: this.y, size: 9, font: this.bold, color: MUTED });
      lines.forEach((line, i) => {
        if (i > 0) this.y -= 11.5;
        this.page.drawText(line, { x: MARGIN + 130, y: this.y, size: 9.5, font: this.font, color: INK });
      });
    }
  }

  async image(bytes: Uint8Array, aspect: number, maxHeight = 360) {
    const embedded = await this.doc.embedJpg(bytes);
    const maxWidth = A4[0] - 2 * MARGIN;
    let width = maxWidth;
    let height = width / aspect;
    if (height > maxHeight) {
      height = maxHeight;
      width = height * aspect;
    }
    this.ensure(height + 8);
    this.y -= height + 4;
    this.page.drawImage(embedded, { x: MARGIN, y: this.y, width, height });
  }
}

export interface PdfImage {
  photoId: string;
  jpeg: Uint8Array;
  aspect: number;
}

export async function buildPdf(report: AssessmentReport, images: PdfImage[]): Promise<Uint8Array> {
  const doc = await PDFDocument.create();
  doc.setTitle(pdfSafe(`ImpactLens assessment - ${report.case.title || report.case.id}`));
  doc.setCreator('ImpactLens inspection UI');
  doc.setProducer('ImpactLens / pdf-lib');
  doc.setSubject('Decision-support vehicle inspection record (not a claim decision)');
  const font = await doc.embedFont(StandardFonts.Helvetica);
  const bold = await doc.embedFont(StandardFonts.HelveticaBold);
  const footer = `ImpactLens ${report.report_type} v${report.report_version} - case ${report.case.id} - generated ${report.generated_at} - decision support only`;
  const w = new Writer(doc, font, bold, footer);

  w.text(report.mode === 'illustrative_demo' ? 'ImpactLens - ILLUSTRATIVE DEMO' : 'ImpactLens inspection assessment', { size: 18, bold: true });
  w.text('Draft record of a human review. Not a claim decision, repair estimate or roadworthiness assessment.', { color: MUTED });
  if (report.notice) w.text(report.notice, { bold: true, color: WARN });
  w.rule();
  w.keyValues([
    ['Case', `${report.case.title || '(untitled)'} - ${report.case.id}`],
    ['Generated', report.generated_at],
    ['Reviewer', report.reviewer.name || '(not entered)'],
    ['Vehicle', [report.vehicle.make, report.vehicle.model, report.vehicle.year].filter(Boolean).join(' ') || '(not entered)'],
    ['Body style / colour', [report.vehicle.body_style, report.vehicle.colour].filter(Boolean).join(' / ') || '-'],
    ['Vehicle notes', report.vehicle.notes || '-'],
  ]);

  w.heading('Review status');
  const s = report.review_summary;
  w.keyValues([
    ['Findings listed', `${s.findings_listed} (per photo; not a count of unique damages)`],
    ['Confirmed', String(s.by_review_status.confirmed)],
    ['Pending review', String(s.by_review_status.pending_review)],
    ['Needs more evidence', String(s.by_review_status.needs_more_evidence)],
    ['Rejected', String(s.by_review_status.rejected)],
    ['Origins', `model output ${s.by_origin.model_output}, human annotation ${s.by_origin.human_annotation}, illustrative ${s.by_origin.illustrative_sample}`],
    ['Photos without model run', String(s.photos_without_model_run)],
  ]);
  if (report.reviewer.notes) {
    w.heading('Reviewer notes', 11);
    w.text(report.reviewer.notes);
  }

  w.heading('Workshop quotes (reviewer-entered, INR)');
  if (report.quotes.entries.length === 0) w.text('No workshop quotes were entered.', { color: MUTED });
  for (const entry of report.quotes.entries) {
    w.text(
      `${entry.finding_id.slice(-8)} - ${DAMAGE_LABELS[entry.damage_type as ReviewDamageType] ?? entry.damage_type} - ${formatInr(entry.quote_inr)} - ${entry.counted ? 'counted' : `not counted (${entry.reason})`}`,
      { indent: 8 },
    );
  }
  w.text(`Provisional partial subtotal: ${formatInr(report.quotes.provisional_subtotal_inr)}`, { bold: true });
  w.text(report.quotes.label, { size: 8.5, color: WARN });

  w.heading('Limitations');
  for (const line of report.disclaimers) w.text(`- ${line}`, { size: 8.5, color: MUTED, indent: 4 });

  for (const photo of report.photos) {
    w.newPage();
    w.text(`Photo ${photo.position}: ${photo.file_name}`, { size: 14, bold: true });
    const image = images.find((i) => i.photoId === photo.photo_id);
    if (image) await w.image(image.jpeg, image.aspect);
    else w.text('Annotated image unavailable: the photo bytes were not attached in this session.', { color: WARN });
    w.keyValues([
      ['Dimensions', `${photo.width} x ${photo.height} px (EXIF-oriented)`],
      ['SHA-256', photo.sha256],
      [
        'Model run',
        photo.model_run
          ? `${photo.model_run.model_id} @ ${photo.model_run.model_revision}; checkpoint ${photo.model_run.checkpoint_sha256.slice(0, 16)}...; score threshold ${photo.model_run.score_threshold}; request ${photo.model_run.request_id}`
          : 'not run on this photo',
      ],
      ['Model outcome', photo.model_run ? photo.model_run.message : '-'],
    ]);
    w.heading('Findings', 11);
    if (photo.findings.length === 0) w.text('No findings recorded for this photo.', { color: MUTED });
    for (const f of photo.findings) {
      w.ensure(60);
      w.rule();
      const score = f.original_prediction ? `, model score ${f.original_prediction.score.toFixed(3)} (uncalibrated)` : '';
      w.text(`#${f.number} ${DAMAGE_LABELS[f.damage_type as ReviewDamageType] ?? f.damage_type} - ${f.origin_label}${score}`, { bold: true });
      if (f.edited_by_reviewer && f.original_prediction) {
        w.text(`Model originally predicted: ${DAMAGE_LABELS[f.original_prediction.damage_type]}`, { color: WARN, indent: 8 });
      }
      w.text(
        `Review: ${REVIEW_STATUS_LABELS[f.review_status]} | Part: ${f.part ?? 'not entered'} | Visible severity: ${SEVERITY_LABELS[f.visible_severity]} | Quote: ${f.quote_inr === null ? '-' : formatInr(f.quote_inr)}${f.same_damage_as ? ` | same damage as ${f.same_damage_as.slice(-8)}` : ''}`,
        { indent: 8 },
      );
      if (f.notes) w.text(`Notes: ${f.notes}`, { indent: 8, color: MUTED });
      if (f.original_prediction?.flags.length) w.text(`Flags: ${f.original_prediction.flags.join(', ')}`, { indent: 8, color: MUTED, size: 8.5 });
    }
  }
  return doc.save();
}
