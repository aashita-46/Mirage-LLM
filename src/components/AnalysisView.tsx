import { motion } from "framer-motion";
import { CheckCircle2, Clock3, Cpu, Download, Layers3, ShieldQuestion } from "lucide-react";
import type { Analysis } from "../types";
import { ScoreGauge } from "./ScoreGauge";
import { TokenHeatmap } from "./TokenHeatmap";

export function AnalysisView({ analysis }: { analysis: Analysis }) {
  const download = () => {
    const blob = new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `mirage-${analysis.id.slice(0,8)}.json`; a.click();
  };
  return <motion.section className="analysis-grid" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}>
    <article className="panel answer-panel">
      <header><div><span className="eyebrow">Primary generation</span><h3>Streamed answer</h3></div><button className="icon-button" onClick={download} title="Download analysis JSON"><Download size={17}/></button></header>
      <p className="answer">{analysis.answer}</p>
      <div className="meta-row"><span><Cpu size={14}/>{analysis.device}</span><span><Clock3 size={14}/>{analysis.latency}s</span><span><CheckCircle2 size={14}/>complete</span></div>
    </article>
    <article className="panel score-panel">
      <header><div><span className="eyebrow">Experimental · uncalibrated</span><h3>Observed risk</h3></div></header>
      <ScoreGauge score={analysis.score}/>
      <div className="score-label">{analysis.score < 25 ? "Lower observed uncertainty" : analysis.score < 50 ? "Moderate uncertainty" : "High disagreement"}</div>
      <div className="breakdown">
        {[["Semantic entropy", analysis.breakdown.semantic], ["Token uncertainty", analysis.breakdown.token], ["Inverse P(True)", analysis.breakdown.inversePTrue]].map(([label, value]) =>
          <div key={String(label)}><span>{label}</span><div><i style={{width: `${Number(value)*100}%`}}/></div><b>{(Number(value)*100).toFixed(0)}%</b></div>)}
      </div>
    </article>
    <article className="panel token-panel">
      <header><div><span className="eyebrow">Token-level lens</span><h3>Confidence heatmap</h3></div><span className="pill">{analysis.tokens.length} tokens</span></header>
      <TokenHeatmap tokens={analysis.tokens}/>
    </article>
    <article className="panel cluster-panel">
      <header><div><span className="eyebrow">Meaning-level disagreement</span><h3>Semantic clusters</h3></div><span className="pill"><Layers3 size={13}/>{analysis.clusters.length} clusters</span></header>
      <div className="cluster-bars">{analysis.clusters.map((c, i) => <div className="cluster" key={c.id}>
        <div className="cluster-head"><b>{c.label}</b><span>{Math.round(c.probability*100)}%</span></div>
        <div className="bar"><motion.i initial={{width: 0}} animate={{width: `${c.probability*100}%`}} style={{background: i ? "#f0a85a" : "#6b65e8"}}/></div>
        <p>{c.representativeAnswer}</p><small>{c.sampleIds.length} sampled answers</small>
      </div>)}</div>
      <div className="entropy"><span>Normalised semantic entropy</span><strong>{analysis.normalisedSemanticEntropy.toFixed(3)}</strong></div>
    </article>
    <article className="panel samples-panel">
      <header><div><span className="eyebrow">Temperature samples</span><h3>Sample explorer</h3></div><span className="pill">{analysis.samples.length}/{analysis.samples.length} analysed</span></header>
      <div className="sample-list">{analysis.samples.map((s, i) => <motion.div className="sample" key={s.id} initial={{opacity:0,x:10}} animate={{opacity:1,x:0}} transition={{delay:i*.06}}>
        <span className="sample-index">0{i+1}</span><p>{s.answer}</p><div><span>C{s.cluster+1}</span><span>{Math.round(s.agreement*100)}% agree</span></div>
      </motion.div>)}</div>
    </article>
    <article className="panel verify-panel">
      <header><div><span className="eyebrow">Model self-verification</span><h3>P(True)</h3></div><ShieldQuestion size={20}/></header>
      <div className="ptrue"><strong>{Math.round(analysis.pTrue*100)}%</strong><div className="bar"><i style={{width:`${analysis.pTrue*100}%`}}/></div></div>
      <p>{analysis.verification}</p><p className="micro">Self-verification is an additional model signal and is not independent ground truth.</p>
    </article>
  </motion.section>;
}
