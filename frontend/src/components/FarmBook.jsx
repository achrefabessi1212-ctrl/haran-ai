import { useEffect, useState } from "react";
import { formatPlace, formatWhen, readFarmChecks } from "../utils/farmBook.js";

export default function FarmBook() {
  const [checks, setChecks] = useState(() => readFarmChecks());

  useEffect(() => {
    const refresh = () => setChecks(readFarmChecks());
    window.addEventListener("focus", refresh);
    window.addEventListener("storage", refresh);
    return () => {
      window.removeEventListener("focus", refresh);
      window.removeEventListener("storage", refresh);
    };
  }, []);

  const sick = checks.filter((item) => item.status === "sick");
  const high = sick.filter((item) => item.severity === "High").length;
  const medium = sick.filter((item) => item.severity === "Medium").length;
  const low = sick.filter((item) => item.severity === "Low").length;

  return (
    <section className="guardian-panel farm-book">
      <div className="book-summary">
        <article className="summary-tile">
          <p>Checks</p>
          <strong>{checks.length}</strong>
        </article>
        <article className="summary-tile high">
          <p>High</p>
          <strong>{high}</strong>
        </article>
        <article className="summary-tile medium">
          <p>Medium</p>
          <strong>{medium}</strong>
        </article>
        <article className="summary-tile low">
          <p>Low</p>
          <strong>{low}</strong>
        </article>
      </div>

      {checks.length === 0 ? (
        <article className="status-card">
          <p className="status-kicker">Farm book</p>
          <h2>No tree checks yet.</h2>
          <p className="prescription">
            Photograph an olive leaf in Grove watch. Each check is saved here with the place, the time, and what to do.
          </p>
        </article>
      ) : (
        <ol className="check-list">
          {checks.map((item) => (
            <li key={item.id} className={`check-card severity-${String(item.severity || "none").toLowerCase()}`}>
              <header className="check-head">
                <span className={`severity-pill severity-${String(item.severity || "none").toLowerCase()}`}>
                  {item.status === "healthy" ? "Healthy" : item.severity || "Check"}
                </span>
                <h2>{item.disease}</h2>
              </header>
              <dl className="check-facts">
                <div>
                  <dt>Checked</dt>
                  <dd>{formatWhen(item.checkedAt)}</dd>
                </div>
                <div>
                  <dt>Check again</dt>
                  <dd>{formatWhen(item.checkAgain)}</dd>
                </div>
                <div>
                  <dt>Location</dt>
                  <dd>{formatPlace(item.lat, item.lng)}</dd>
                </div>
                {item.damagePercent != null && (
                  <div>
                    <dt>Leaf area</dt>
                    <dd>{item.damagePercent}%</dd>
                  </div>
                )}
              </dl>
              <p className="solution-line">{item.solution}</p>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
