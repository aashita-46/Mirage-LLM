import { motion } from "framer-motion";

export function ScoreGauge({ score, compact = false }: { score: number; compact?: boolean }) {
  const radius = 76, circumference = Math.PI * radius;
  const colour = score < 25 ? "#58d6c7" : score < 50 ? "#f5b94c" : score < 75 ? "#ff7b55" : "#ff4d68";
  return <div className={`gauge ${compact ? "compact" : ""}`} aria-label={`MirageScore ${score} out of 100`}>
    <svg viewBox="0 0 180 105" role="img">
      <path d="M 14 92 A 76 76 0 0 1 166 92" pathLength="1" className="gauge-track" />
      <motion.path d="M 14 92 A 76 76 0 0 1 166 92" pathLength="1"
        className="gauge-value" style={{ stroke: colour }}
        initial={{ pathLength: 0 }} animate={{ pathLength: score / 100 }} transition={{ duration: .9, ease: "easeOut" }} />
    </svg>
    <div className="gauge-copy"><strong>{score.toFixed(1)}</strong><span>MirageScore</span></div>
  </div>;
}
