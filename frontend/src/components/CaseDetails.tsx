import type { InspectionCase, VehicleContext } from '../lib/types';

interface Props {
  inspection: InspectionCase;
  onDetails: (patch: Partial<Pick<InspectionCase, 'title' | 'reviewer_name' | 'reviewer_notes'>>) => void;
  onVehicle: (patch: Partial<VehicleContext>) => void;
}

const BODY_STYLES = ['', 'Hatchback', 'Sedan', 'SUV', 'MUV / MPV', 'Pickup', 'Van', 'Coupe', 'Convertible', 'Two-wheeler', 'Three-wheeler', 'Other'];

export function CaseDetails({ inspection, onDetails, onVehicle }: Props) {
  const v = inspection.vehicle;
  return (
    <section className="panel" aria-labelledby="case-heading">
      <h2 id="case-heading">Case and vehicle context</h2>
      <p className="help">Optional. Registration number and owner details are not collected and are not read from photos.</p>
      <div className="field-grid">
        <label htmlFor="case-title">Case title</label>
        <input id="case-title" value={inspection.title} maxLength={120} onChange={(e) => onDetails({ title: e.target.value })} placeholder="e.g. Workshop intake, front-left impact" />

        <label htmlFor="vehicle-make">Make</label>
        <input id="vehicle-make" value={v.make} maxLength={60} autoComplete="off" onChange={(e) => onVehicle({ make: e.target.value })} />

        <label htmlFor="vehicle-model">Model</label>
        <input id="vehicle-model" value={v.model} maxLength={60} autoComplete="off" onChange={(e) => onVehicle({ model: e.target.value })} />

        <label htmlFor="vehicle-year">Year</label>
        <input
          id="vehicle-year"
          value={v.year}
          inputMode="numeric"
          pattern="(19|20)[0-9]{2}"
          maxLength={4}
          autoComplete="off"
          onChange={(e) => onVehicle({ year: e.target.value.replace(/\D/g, '').slice(0, 4) })}
        />

        <label htmlFor="vehicle-body">Body style</label>
        <select id="vehicle-body" value={v.body_style} onChange={(e) => onVehicle({ body_style: e.target.value })}>
          {BODY_STYLES.map((style) => (
            <option key={style} value={style}>
              {style || 'Not specified'}
            </option>
          ))}
        </select>

        <label htmlFor="vehicle-colour">Colour</label>
        <input id="vehicle-colour" value={v.colour} maxLength={40} onChange={(e) => onVehicle({ colour: e.target.value })} />

        <label htmlFor="vehicle-notes">Vehicle notes</label>
        <textarea id="vehicle-notes" rows={2} maxLength={1000} value={v.notes} onChange={(e) => onVehicle({ notes: e.target.value })} />

        <label htmlFor="reviewer-name">Reviewer</label>
        <input id="reviewer-name" value={inspection.reviewer_name} maxLength={80} autoComplete="name" onChange={(e) => onDetails({ reviewer_name: e.target.value })} />

        <label htmlFor="reviewer-notes">Reviewer notes</label>
        <textarea id="reviewer-notes" rows={3} maxLength={4000} value={inspection.reviewer_notes} onChange={(e) => onDetails({ reviewer_notes: e.target.value })} />
      </div>
    </section>
  );
}
