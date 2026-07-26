import { useState } from "react";
import type { TokenRisk } from "../types";

export function TokenHeatmap({ tokens }: { tokens: TokenRisk[] }) {
  const [highOnly, setHighOnly] = useState(false);
  const [metric, setMetric] = useState<"weightedRisk" | "normalisedUncertainty">("weightedRisk");
  return <div>
    <div className="panel-tools">
      <div className="segmented">
        <button className={metric === "weightedRisk" ? "active" : ""} onClick={() => setMetric("weightedRisk")}>Weighted risk</button>
        <button className={metric === "normalisedUncertainty" ? "active" : ""} onClick={() => setMetric("normalisedUncertainty")}>Raw uncertainty</button>
      </div>
      <label className="switch"><input type="checkbox" checked={highOnly} onChange={e => setHighOnly(e.target.checked)} /> High risk only</label>
    </div>
    <div className="tokens" aria-label="Token uncertainty heatmap">
      {tokens.filter(t => !highOnly || t[metric] > .58).map(t => {
        const v = t[metric], hue = 175 - v * 175;
        return <span key={t.id} className="token" style={{ background: `hsla(${hue}, 78%, 54%, ${.09 + v * .35})`, borderColor: `hsla(${hue}, 75%, 58%, ${.25 + v * .45})` }}
          title={`${t.text.trim()} · ${metric}: ${v.toFixed(2)} · ${t.category} × ${t.importanceWeight}`}>
          {t.text}
        </span>;
      })}
    </div>
    <p className="micro">Low token confidence can indicate linguistic uncertainty, but it is not by itself proof of hallucination.</p>
  </div>;
}
