import { useId, type CSSProperties } from 'react';

import { DAMAGE_COLOURS } from '../lib/annotate';
import { isEditedPrediction, originLabel, type FindingPatch } from '../lib/findings';
import {
  DAMAGE_LABELS,
  FLAG_DESCRIPTIONS,
  MODEL_DAMAGE_TYPES,
  REVIEW_DAMAGE_TYPES,
  REVIEW_STATUS_LABELS,
  SEVERITY_LABELS,
  type Finding,
  type InspectionCase,
  type ReviewDamageType,
  type ReviewStatus,
  type VisibleSeverity,
} from '../lib/types';

const COMMON_PARTS = [
  'Front bumper',
  'Rear bumper',
  'Bonnet',
  'Boot lid / tailgate',
  'Roof',
  'Front left door',
  'Front right door',
  'Rear left door',
  'Rear right door',
  'Front left fender',
  'Front right fender',
  'Rear left quarter panel',
  'Rear right quarter panel',
  'Windscreen',
  'Rear windscreen',
  'Side window',
  'Left headlamp',
  'Right headlamp',
  'Left tail lamp',
  'Right tail lamp',
  'Left mirror',
  'Right mirror',
  'Front left tyre',
  'Front right tyre',
  'Rear left tyre',
  'Rear right tyre',
];

interface Props {
  inspection: InspectionCase;
  photoId: string;
  numbers: Map<string, number>;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onUpdate: (id: string, patch: FindingPatch) => void;
  onDelete: (id: string) => void;
}

function originClass(finding: Finding): string {
  if (finding.origin === 'illustrative_sample') return 'badge badge-sample';
  if (finding.origin === 'human_annotation') return 'badge badge-human';
  return isEditedPrediction(finding) ? 'badge badge-edited' : 'badge badge-model';
}

export function FindingsPanel({ inspection, photoId, numbers, selectedId, onSelect, onUpdate, onDelete }: Props) {
  const findings = inspection.findings.filter((f) => f.photo_id === photoId);
  const photo = inspection.photos.find((p) => p.id === photoId);

  return (
    <section className="panel findings" aria-labelledby="findings-heading">
      <h2 id="findings-heading">Findings for this photo</h2>
      {photo?.model_run && (
        <p className={`notice ${photo.model_run.outcome === 'detections_returned' ? 'notice-info' : 'notice-neutral'}`} data-testid="model-outcome">
          {photo.model_run.message}
          {photo.model_run.warnings.length > 0 && <span className="small"> Warnings: {photo.model_run.warnings.join(', ')}.</span>}
        </p>
      )}
      {findings.length === 0 ? (
        <p className="empty">
          No findings yet. Run the model on this photo, or draw a region yourself. An empty list is not evidence that the vehicle is
          undamaged.
        </p>
      ) : (
        <ol className="finding-list">
          {findings.map((finding) => {
            const number = numbers.get(finding.id) ?? 0;
            const selected = finding.id === selectedId;
            return (
              <li key={finding.id} className={selected ? 'is-selected' : undefined}>
                <button
                  type="button"
                  className="finding-row"
                  aria-expanded={selected}
                  aria-controls={`editor-${finding.id}`}
                  onClick={() => onSelect(selected ? null : finding.id)}
                  data-testid={`finding-${number}`}
                >
                  <span className="swatch" aria-hidden="true" style={{ '--swatch': DAMAGE_COLOURS[finding.damage_type] } as CSSProperties} />
                  <span className="finding-title">
                    #{number} {DAMAGE_LABELS[finding.damage_type]}
                  </span>
                  <span className={originClass(finding)}>{originLabel(finding)}</span>
                  <span className={`badge status-${finding.review_status}`}>{REVIEW_STATUS_LABELS[finding.review_status]}</span>
                  {finding.original && <span className="score">model score {finding.original.score.toFixed(2)}</span>}
                </button>
                {selected && (
                  <FindingEditor
                    finding={finding}
                    number={number}
                    inspection={inspection}
                    numbers={numbers}
                    onUpdate={(patch) => onUpdate(finding.id, patch)}
                    onDelete={() => onDelete(finding.id)}
                  />
                )}
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

interface EditorProps {
  finding: Finding;
  number: number;
  inspection: InspectionCase;
  numbers: Map<string, number>;
  onUpdate: (patch: FindingPatch) => void;
  onDelete: () => void;
}

function FindingEditor({ finding, number, inspection, numbers, onUpdate, onDelete }: EditorProps) {
  const id = useId();
  const others = inspection.findings.filter((f) => f.id !== finding.id && f.same_damage_as !== finding.id);
  const photoPosition = (photoId: string) => inspection.photos.findIndex((p) => p.id === photoId) + 1;
  const original = finding.original;

  return (
    <form className="editor" id={`editor-${finding.id}`} aria-label={`Edit finding ${number}`} onSubmit={(e) => e.preventDefault()}>
      {original && (
        <div className="original" data-testid="original-prediction">
          <p className="small">
            <strong>Model prediction (kept unchanged):</strong> {DAMAGE_LABELS[original.damage_type]}, score {original.score.toFixed(3)}{' '}
            (uncalibrated, not a probability), model <code>{original.model_id}</code>.
          </p>
          {isEditedPrediction(finding) && (
            <p className="small warn-text" data-testid="edited-note">
              The reviewer changed the label from {DAMAGE_LABELS[original.damage_type]} to {DAMAGE_LABELS[finding.damage_type]}.
            </p>
          )}
          {original.flags.length > 0 && (
            <ul className="small flags">
              {original.flags.map((flag) => (
                <li key={flag}>
                  <code>{flag}</code>: {FLAG_DESCRIPTIONS[flag]}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
      {finding.origin === 'illustrative_sample' && <p className="small warn-text">Illustrative sample: drawn by hand for the demo, not produced by a model.</p>}

      <div className="field-grid">
        <label htmlFor={`${id}-type`}>Damage type</label>
        <select id={`${id}-type`} value={finding.damage_type} onChange={(e) => onUpdate({ damage_type: e.target.value as ReviewDamageType })}>
          {REVIEW_DAMAGE_TYPES.map((type) => (
            <option key={type} value={type}>
              {DAMAGE_LABELS[type]}
              {(MODEL_DAMAGE_TYPES as readonly string[]).includes(type) ? '' : ' (reviewer only)'}
            </option>
          ))}
        </select>

        <label htmlFor={`${id}-status`}>Review status</label>
        <select id={`${id}-status`} value={finding.review_status} onChange={(e) => onUpdate({ review_status: e.target.value as ReviewStatus })}>
          {(Object.keys(REVIEW_STATUS_LABELS) as ReviewStatus[]).map((status) => (
            <option key={status} value={status}>
              {REVIEW_STATUS_LABELS[status]}
            </option>
          ))}
        </select>

        <label htmlFor={`${id}-part`}>Affected part</label>
        <div>
          <input
            id={`${id}-part`}
            list={`${id}-parts`}
            value={finding.part ?? ''}
            maxLength={80}
            placeholder="Reviewer entry, e.g. Rear left door"
            aria-describedby={`${id}-part-help`}
            onChange={(e) => onUpdate({ part: e.target.value.trim() ? e.target.value : null })}
          />
          <datalist id={`${id}-parts`}>
            {COMMON_PARTS.map((part) => (
              <option key={part} value={part} />
            ))}
          </datalist>
          <p className="help" id={`${id}-part-help`}>
            Not predicted: the baseline has no validated part model.
          </p>
        </div>

        <label htmlFor={`${id}-severity`}>Visible severity</label>
        <div>
          <select
            id={`${id}-severity`}
            value={finding.severity}
            aria-describedby={`${id}-severity-help`}
            onChange={(e) => onUpdate({ severity: e.target.value as VisibleSeverity })}
          >
            {(Object.keys(SEVERITY_LABELS) as VisibleSeverity[]).map((severity) => (
              <option key={severity} value={severity}>
                {SEVERITY_LABELS[severity]}
              </option>
            ))}
          </select>
          <p className="help" id={`${id}-severity-help`}>
            Your visual estimate of what the photo shows. Not predicted, and not derived from region size.
          </p>
        </div>

        <label htmlFor={`${id}-same`}>Same damage as</label>
        <div>
          <select
            id={`${id}-same`}
            value={finding.same_damage_as ?? ''}
            aria-describedby={`${id}-same-help`}
            onChange={(e) => onUpdate({ same_damage_as: e.target.value || null })}
          >
            <option value="">Not linked</option>
            {others.map((other) => (
              <option key={other.id} value={other.id}>
                Photo {photoPosition(other.photo_id)}, #{numbers.get(other.id)} {DAMAGE_LABELS[other.damage_type]}
              </option>
            ))}
          </select>
          <p className="help" id={`${id}-same-help`}>
            Link findings that show one physical damage from different views, so a quote is not counted twice.
          </p>
        </div>

        <label htmlFor={`${id}-quote`}>Workshop quote (INR)</label>
        <div>
          <input
            id={`${id}-quote`}
            type="number"
            inputMode="decimal"
            min={0}
            max={100000000}
            step="0.01"
            disabled={finding.same_damage_as !== null}
            value={finding.quote_inr ?? ''}
            aria-describedby={`${id}-quote-help`}
            onChange={(e) => {
              const value = e.target.value === '' ? null : Number(e.target.value);
              onUpdate({ quote_inr: value !== null && Number.isFinite(value) && value >= 0 ? value : null });
            }}
          />
          <p className="help" id={`${id}-quote-help`}>
            {finding.same_damage_as
              ? 'Recorded on the linked finding.'
              : 'An actual quote from a workshop, entered by you. ImpactLens never computes prices.'}
          </p>
        </div>

        <label htmlFor={`${id}-notes`}>Notes on this finding</label>
        <textarea id={`${id}-notes`} rows={3} maxLength={2000} value={finding.notes} onChange={(e) => onUpdate({ notes: e.target.value })} />
      </div>

      <div className="editor-actions">
        <button type="button" className="btn btn-small" onClick={() => onUpdate({ review_status: 'confirmed' })}>
          Confirm
        </button>
        <button type="button" className="btn btn-small" onClick={() => onUpdate({ review_status: 'rejected' })}>
          Reject
        </button>
        {finding.origin === 'human_annotation' && (
          <button type="button" className="btn btn-small btn-danger" onClick={onDelete}>
            Delete annotation
          </button>
        )}
      </div>
    </form>
  );
}
