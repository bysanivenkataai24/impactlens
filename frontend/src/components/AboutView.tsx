import { DAMAGE_LABELS, MODEL_DAMAGE_TYPES } from '../lib/types';

export function AboutView({ sourceUrl }: { sourceUrl: string | null }) {
  return (
    <div className="view">
      <section className="panel prose" aria-labelledby="about-heading">
        <h2 id="about-heading">What ImpactLens does and does not do</h2>
        <p>
          ImpactLens suggests visible damage regions in vehicle photos so that an inspector can review, correct and document them. It is
          decision support. Every model region starts as <em>pending review</em>.
        </p>
        <h3>Supported damage types</h3>
        <p>
          {MODEL_DAMAGE_TYPES.map((t) => DAMAGE_LABELS[t]).join(', ')}. Reviewers can also label a region <em>Other / uncertain</em>; the
          model never returns that label. Damage outside these types is not detected.
        </p>
        <h3>Not determined from a photo</h3>
        <ul>
          <li>Mechanical condition, roadworthiness, hidden or structural damage.</li>
          <li>Fault, claim eligibility, approval or denial.</li>
          <li>Repair cost. Workshop quotes are entered by the reviewer; nothing is priced automatically.</li>
          <li>Affected part and severity: the baseline model has no validated part or severity model, so these are reviewer entries.</li>
        </ul>
        <h3>Reading the output</h3>
        <ul>
          <li>The model score is uncalibrated. It ranks regions; it is not the probability that damage is present.</li>
          <li>An empty result means no supported region was returned at that threshold. It does not mean the vehicle is undamaged.</li>
          <li>Several photos can show the same damage. Findings are per photo and are not counted as unique damages.</li>
          <li>Findings are labelled by origin: model output, model output edited by a reviewer, human annotation, or illustrative sample.</li>
        </ul>
        <h3>Privacy</h3>
        <ul>
          <li>Photos stay in this tab's memory and are sent only when you press “Run model”, only to the configured service.</li>
          <li>The service decodes photos in memory, discards their metadata (including GPS) and does not store them.</li>
          <li>Exported annotated images are re-encoded without metadata. Registration numbers and faces are not redacted automatically.</li>
        </ul>
        <h3>Source code and licence</h3>
        <p>
          ImpactLens is licensed under the GNU AGPL-3.0, as is the Ultralytics library its model runs on.
          {sourceUrl ? (
            <>
              {' '}
              Source for this deployment:{' '}
              <a href={sourceUrl} rel="noopener noreferrer" target="_blank">
                {sourceUrl}
              </a>
              .
            </>
          ) : (
            ' The operator of this deployment must make its source available to users (AGPL-3.0 section 13).'
          )}
        </p>
      </section>
    </div>
  );
}
