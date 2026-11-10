import { useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent } from 'react';

import { DAMAGE_COLOURS, photoSource } from '../lib/annotate';
import { originLabel } from '../lib/findings';
import { normalizedPoint, toSvgPoints } from '../lib/geometry';
import { DAMAGE_LABELS, REVIEW_STATUS_LABELS, type Finding, type Photo, type Point } from '../lib/types';

interface Props {
  photo: Photo;
  position: number;
  findings: Finding[];
  numbers: Map<string, number>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  showOverlays: boolean;
  showRejected: boolean;
  drawing: boolean;
  draft: Point[];
  onAddPoint: (point: Point) => void;
  onCancelDrawing: () => void;
}

export function PhotoViewer(props: Props) {
  const { photo, findings, numbers, selectedId, onSelect, showOverlays, showRejected, drawing, draft } = props;
  const surface = useRef<SVGSVGElement>(null);
  // Keyed by photo id, so switching photos never carries a stale result over.
  const [mismatchFor, setMismatchFor] = useState<string | null>(null);
  const orientationProblem = mismatchFor === photo.id;
  const source = photoSource(photo);

  const visible = findings.filter((f) => showRejected || f.review_status !== 'rejected');
  const unit = Math.max(photo.width, photo.height) / 1000;
  const style = { '--ar': `${photo.width} / ${photo.height}`, '--ar-number': photo.width / photo.height } as CSSProperties;

  function handlePointer(event: PointerEvent<SVGSVGElement>) {
    if (!drawing || !surface.current || event.button !== 0) return;
    event.preventDefault();
    props.onAddPoint(normalizedPoint(event.clientX, event.clientY, surface.current.getBoundingClientRect()));
  }

  function handleKey(event: KeyboardEvent<SVGSVGElement>) {
    if (drawing && event.key === 'Escape') props.onCancelDrawing();
  }

  if (!source) {
    return (
      <div className="stage-missing" role="note">
        <p>
          <strong>Photo not stored.</strong> Saved cases keep findings and the photo's SHA-256, but not its bytes. Re-attach{' '}
          <code>{photo.file_name}</code> below to view it.
        </p>
      </div>
    );
  }

  return (
    <div className="stage-wrap">
      <div className={`stage${drawing ? ' is-drawing' : ''}`} style={style} data-testid="stage">
        <img
          className="stage-img"
          src={source}
          alt={`Evidence ${props.position}: ${photo.file_name}. Damage regions are listed in the findings panel.`}
          draggable={false}
          onLoad={(event) => {
            const image = event.currentTarget;
            // Overlays assume the browser orients the photo exactly as the service did.
            setMismatchFor(image.naturalWidth !== photo.width || image.naturalHeight !== photo.height ? photo.id : null);
          }}
        />
        {orientationProblem ? null : (
          <svg
            ref={surface}
            className="overlay"
            data-testid="overlay"
            viewBox={`0 0 ${photo.width} ${photo.height}`}
            preserveAspectRatio="none"
            onPointerDown={handlePointer}
            onKeyDown={handleKey}
            aria-hidden={drawing ? undefined : true}
            role={drawing ? 'img' : undefined}
            aria-label={drawing ? `Drawing a region: ${draft.length} point(s) placed` : undefined}
          >
            {showOverlays &&
              visible.map((finding) => {
                const selected = finding.id === selectedId;
                const dimmed = selectedId !== null && !selected;
                const colour = DAMAGE_COLOURS[finding.damage_type];
                const number = numbers.get(finding.id) ?? 0;
                const [x0, y0] = finding.region.bbox;
                return (
                  <g
                    key={finding.id}
                    className={`region${selected ? ' is-selected' : ''}${dimmed ? ' is-dimmed' : ''}${finding.review_status === 'rejected' ? ' is-rejected' : ''}`}
                    data-finding-id={finding.id}
                    onPointerDown={(event) => {
                      if (drawing) return;
                      event.stopPropagation();
                      onSelect(selected ? null : finding.id);
                    }}
                  >
                    <title>{`#${number} ${DAMAGE_LABELS[finding.damage_type]} - ${originLabel(finding)} - ${REVIEW_STATUS_LABELS[finding.review_status]}`}</title>
                    <polygon className="region-halo" points={toSvgPoints(finding.region.polygon, photo.width, photo.height)} vectorEffect="non-scaling-stroke" />
                    <polygon
                      className="region-shape"
                      points={toSvgPoints(finding.region.polygon, photo.width, photo.height)}
                      fill={colour}
                      stroke={colour}
                      vectorEffect="non-scaling-stroke"
                      data-testid={`region-${number}`}
                    />
                    <text
                      className="region-label"
                      x={x0 * photo.width + 4 * unit}
                      y={Math.max(y0 * photo.height - 6 * unit, 22 * unit)}
                      fontSize={22 * unit}
                      fill={colour}
                    >
                      {`#${number} ${DAMAGE_LABELS[finding.damage_type]}`}
                    </text>
                  </g>
                );
              })}
            {drawing && draft.length > 0 && (
              <g className="draft">
                <polyline points={toSvgPoints(draft, photo.width, photo.height)} vectorEffect="non-scaling-stroke" />
                {draft.map(([x, y], i) => (
                  <circle key={i} cx={x * photo.width} cy={y * photo.height} r={6 * unit} />
                ))}
              </g>
            )}
          </svg>
        )}
      </div>
      {orientationProblem && (
        <p className="notice notice-warn" role="alert">
          This browser displays the photo with different dimensions than the service uses, so overlays would be misaligned and are
          hidden. Use a current version of Chrome, Edge, Firefox or Safari.
        </p>
      )}
    </div>
  );
}
