// Illustrative demo mode. Separate from live inspection by construction:
// the case has mode 'illustrative_demo', its photo is a bundled drawing, every
// finding has origin 'illustrative_sample', and demo photos cannot be sent to
// the service. Nothing here is ever used to fill in for a failed request.

import { regionFromPolygon } from './geometry';
import { newId } from './photos';
import type { Finding, InspectionCase, Point } from './types';

export const DEMO_IMAGE_URL = './samples/illustrative-hatchback.svg';
const DEMO_SHA256 = 'illustrative-drawing-no-photo-bytes'.padEnd(64, '0');

function sampleFinding(photoId: string, damage: Finding['damage_type'], polygon: Point[], notes: string): Finding {
  const region = regionFromPolygon(polygon);
  if (!region) throw new Error('invalid demo polygon');
  const timestamp = new Date().toISOString();
  return {
    id: newId('sample'),
    photo_id: photoId,
    origin: 'illustrative_sample',
    original: null,
    damage_type: damage,
    region,
    part: null,
    severity: 'unknown',
    review_status: 'pending_review',
    notes,
    quote_inr: null,
    same_damage_as: null,
    created_at: timestamp,
    updated_at: timestamp,
  };
}

const ellipse = (cx: number, cy: number, rx: number, ry: number, n = 20): Point[] =>
  Array.from({ length: n }, (_, i) => {
    const a = (2 * Math.PI * i) / n;
    return [(cx + rx * Math.cos(a)) / 1200, (cy + ry * Math.sin(a)) / 800] as Point;
  });

export function createDemoCase(): InspectionCase {
  const photoId = newId('photo');
  const timestamp = new Date().toISOString();
  return {
    id: newId('case'),
    mode: 'illustrative_demo',
    title: 'Illustrative demo',
    created_at: timestamp,
    updated_at: timestamp,
    vehicle: { make: '', model: '', year: '', body_style: 'Hatchback (illustration)', colour: 'Blue', notes: '' },
    reviewer_name: '',
    reviewer_notes: '',
    photos: [
      {
        id: photoId,
        file_name: 'illustrative-hatchback.svg',
        media_type: 'image/svg+xml',
        bytes: 0,
        sha256: DEMO_SHA256,
        width: 1200,
        height: 800,
        object_url: null,
        sample_url: DEMO_IMAGE_URL,
        model_run: null,
      },
    ],
    findings: [
      sampleFinding(photoId, 'dent', ellipse(740, 500, 74, 40), 'Sample finding drawn by hand to show the review workflow.'),
      sampleFinding(
        photoId,
        'scratch',
        [
          [415 / 1200, 508 / 800],
          [470 / 1200, 498 / 800],
          [592 / 1200, 490 / 800],
          [592 / 1200, 512 / 800],
          [470 / 1200, 528 / 800],
          [418 / 1200, 530 / 800],
        ],
        'Sample finding. Try editing its type, part, severity and review status.',
      ),
      sampleFinding(
        photoId,
        'lamp_broken',
        [
          [148 / 1200, 444 / 800],
          [194 / 1200, 440 / 800],
          [194 / 1200, 492 / 800],
          [148 / 1200, 496 / 800],
        ],
        'Sample finding on an illustrated rear lamp.',
      ),
    ],
  };
}
