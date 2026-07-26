import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, ArrowRight, Atom, BarChart3, BookOpen, BrainCircuit, ChevronDown, CircleGauge, Database, FlaskConical, Github, Menu, Play, RefreshCw, SearchCheck, ShieldAlert, Sparkles, TestTube2, X } from "lucide-react";
import { api, type StressVariant } from "./lib/api";
import type { Analysis, Bench } from "./types";
import { AnalysisView } from "./components/AnalysisView";
import { ScoreGauge } from "./components/ScoreGauge";

const examples = [
  "What is the capital of Australia?",
  "Who discovered penicillin, and in which year?",
  "Which treaty formally ended the First World War?",
  "What is the maximum recommended daily dose of paracetamol for a healthy adult?",
  "If every raven is black, does seeing a black bird prove it is a raven?",
];

function Nav({ online }: { online: boolean }) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const close = () => setOpen(false);
    window.addEventListener("resize", close);
    return () => window.removeEventListener("resize", close);
  }, []);
  return <nav><a className="brand" href="#"><span className="brand-mark"/><b>MIRAGE</b><em>LABS</em></a>
    <button className="menu" onClick={()=>setOpen(!open)} aria-label="Toggle navigation" aria-expanded={open} aria-controls="primary-navigation">{open?<X/>:<Menu/>}</button>
    <div id="primary-navigation" className={open ? "nav-links open" : "nav-links"}>
      <a onClick={()=>setOpen(false)} href="#playground">Playground</a><a onClick={()=>setOpen(false)} href="#stress">Stress test</a><a onClick={()=>setOpen(false)} href="#bench">Mirage Bench</a><a onClick={()=>setOpen(false)} href="#method">Methodology</a>
    </div>
    <button className="status-button" onClick={()=>(document.getElementById("system") as HTMLDialogElement | null)?.showModal()}>
      {online ? "Cached demo" : "Backend disconnected"}<ChevronDown size={13}/>
    </button>
  </nav>;
}

function Hero({ launch }: { launch: () => void }) {
  return <header className="hero">
    <motion.div className="hero-copy" initial={{opacity:0,y:24}} animate={{opacity:1,y:0}}>
      <div className="overline"><Sparkles size={14}/> LLM UNCERTAINTY, MADE VISIBLE</div>
      <h1>See where an AI answer starts to become an <span>illusion.</span></h1>
      <p>Mirage analyses token uncertainty, semantic disagreement, self-verification, and paraphrase stability—then turns them into an inspectable risk signal.</p>
      <div className="hero-actions"><button className="primary" onClick={launch}>Launch live analysis <ArrowRight size={17}/></button><a className="secondary" href="#bench">Explore Mirage Bench</a></div>
      <div className="trust-row"><span><i/> No fabricated metrics</span><span><i/> Reproducible metadata</span><span><i/> Graceful CPU fallback</span></div>
    </motion.div>
    <motion.div className="hero-visual" initial={{opacity:0,scale:.96}} animate={{opacity:1,scale:1}} transition={{delay:.15}}>
      <div className="visual-top"><span><i/> ANALYSIS · CACHED DEMONSTRATION</span><span>CANBERRA_01</span></div>
      <div className="visual-body"><div className="mini-answer"><small>PRIMARY ANSWER</small><p>Australia's capital is <mark>Canberra.</mark> It was selected as a compromise between <mark className="medium">Sydney</mark> and <mark className="medium">Melbourne.</mark></p></div><ScoreGauge score={18.4} compact/></div>
      <div className="visual-clusters"><div><span>C1 · Canberra</span><b>5 samples</b><i style={{width:"83%"}}/></div><div><span>C2 · Other</span><b>1 sample</b><i style={{width:"17%"}}/></div></div>
      <div className="visual-foot"><Activity size={15}/><span>Stable meaning across 5 of 6 samples</span><b>LOW OBSERVED UNCERTAINTY</b></div>
    </motion.div>
  </header>;
}

const signals = [
  [CircleGauge, "Token uncertainty", "Surprisal and predictive entropy reveal locally uncertain generation steps."],
  [BrainCircuit, "Semantic entropy", "Sampled answers are clustered by meaning to expose answer-level disagreement."],
  [SearchCheck, "Self-verification", "The model estimates P(True), exposed as a fallible signal—not ground truth."],
  [ShieldAlert, "Prompt stability", "Controlled perturbations reveal fragile answers, flips, and contradictions."],
] as const;

function Playground({ onAnalysis }: { onAnalysis: (a: Analysis)=>void }) {
  const [question, setQuestion] = useState(examples[0]), [domain,setDomain]=useState("General knowledge");
  const [count,setCount]=useState(6), [temperature,setTemperature]=useState(.7), [loading,setLoading]=useState(false);
  const [analysis,setAnalysis]=useState<Analysis|null>(()=>{try{return JSON.parse(localStorage.getItem("mirage-analysis")||"null")}catch{return null}});
  const [error,setError]=useState("");
  const run = async () => { setLoading(true); setError(""); try { const a=await api.analyse(question,domain,count,temperature); setAnalysis(a); onAnalysis(a); localStorage.setItem("mirage-analysis",JSON.stringify(a)); } catch(e){setError(e instanceof Error?e.message:"Analysis failed")} finally{setLoading(false)} };
  return <section id="playground" className="section playground">
    <div className="section-heading"><div><span className="eyebrow">01 · LIVE PLAYGROUND</span><h2>Interrogate an answer, not just the model.</h2></div><p>Watch uncertainty accumulate across tokens, meanings, and self-verification signals.</p></div>
    <div className="prompt-shell">
      <textarea value={question} onChange={e=>setQuestion(e.target.value)} maxLength={2000} aria-label="Question to analyse"/>
      <div className="prompt-suggestions">{examples.slice(0,3).map((x,i)=><button key={x} onClick={()=>setQuestion(x)}>0{i+1} {x.slice(0,34)}…</button>)}</div>
      <div className="controls">
        <label>Domain<select value={domain} onChange={e=>setDomain(e.target.value)}><option>General knowledge</option><option>Medical</option><option>Legal</option><option>Finance</option><option>Science</option></select></label>
        <label>Samples<select value={count} onChange={e=>setCount(+e.target.value)}>{[4,6,8,10].map(n=><option key={n}>{n}</option>)}</select></label>
        <label className="range">Temperature <span>{temperature.toFixed(1)}</span><input type="range" min="0" max="1.2" step=".1" value={temperature} onChange={e=>setTemperature(+e.target.value)}/></label>
        <button className="primary run" onClick={run} disabled={loading||question.trim().length<3}>{loading?<><RefreshCw className="spin" size={17}/> Analysing</>:<><Play size={16}/> Run analysis</>}</button>
      </div>
    </div>
    {error && <div className="error">{error} <button onClick={run}>Retry</button></div>}
    {analysis ? <AnalysisView analysis={analysis}/> : <div className="empty"><Atom/><h3>Your uncertainty map will appear here.</h3><p>Run an analysis to stream an answer and compare sampled meanings.</p></div>}
  </section>;
}

function Stress({ latest }: { latest: Analysis|null }) {
  const [result,setResult]=useState<{variants:StressVariant[];stability:number;instability:number;summary:string}|null>(null);
  const [loading,setLoading]=useState(false);
  const [types,setTypes]=useState(["neutral","formal","distractor","leading","negation"]);
  const toggle=(x:string)=>setTypes(v=>v.includes(x)?v.filter(t=>t!==x):[...v,x]);
  const run=async()=>{if(!latest)return;setLoading(true);try{setResult(await api.stress(latest.question,types))}finally{setLoading(false)}};
  return <section id="stress" className="section">
    <div className="section-heading"><div><span className="eyebrow">02 · ADVERSARIAL STRESS TEST</span><h2>Does the answer survive a change in wording?</h2></div><p>Controlled prompt variants measure fragility without pretending adversarial prompts are equivalent.</p></div>
    <div className="stress-layout"><div className="panel stress-config"><h3>Perturbation suite</h3><p>{latest?.question||"Run a playground analysis first."}</p><div className="chips">{["neutral","formal","conversational","distractor","leading","ambiguity","negation"].map(t=><button className={types.includes(t)?"selected":""} onClick={()=>toggle(t)} key={t}>{t}</button>)}</div><button className="primary" onClick={run} disabled={!latest||loading}>{loading?"Testing variants…":"Run stability test"} <TestTube2 size={17}/></button><small>Negation and leading premise are labelled adversarial variants.</small></div>
      <div className="panel stress-results">{result?<><div className="stability"><div><span>ANSWER STABILITY</span><strong>{Math.round(result.stability*100)}%</strong></div><div><span>INSTABILITY</span><strong>{Math.round(result.instability*100)}%</strong></div></div><p className="summary">{result.summary}</p><div className="variant-list">{result.variants.map(v=><div key={v.id}><span className={v.adversarial?"badge warning":"badge"}>{v.type}</span><p>{v.question}</p><b>{v.score}</b><small>{v.relation}</small></div>)}</div></>:<div className="empty compact"><ShieldAlert/><h3>No variants tested yet.</h3><p>Results are computed from the latest analysis.</p></div>}</div></div>
  </section>;
}

function BenchSection() {
  const [bench,setBench]=useState<Bench|null>(null),[loading,setLoading]=useState(false),[count,setCount]=useState(12);
  const run=async()=>{setLoading(true);try{setBench(await api.bench(count))}finally{setLoading(false)}};
  return <section id="bench" className="section">
    <div className="section-heading"><div><span className="eyebrow">03 · MIRAGE BENCH</span><h2>Evaluate the evaluator.</h2></div><p>Run a legally included reference set. Metrics appear only after they are actually computed.</p></div>
    <div className="bench-toolbar panel"><div><Database/><span><b>Mirage curated demo</b><small>Reference-scored · deterministic risk samples</small></span></div><label>Questions<select value={count} onChange={e=>setCount(+e.target.value)}><option>8</option><option>12</option><option>16</option></select></label><button className="primary" onClick={run} disabled={loading}>{loading?"Computing…":"Start benchmark"} <BarChart3 size={17}/></button></div>
    {bench?<div className="bench-grid"><div className="metrics">{[["AUROC",bench.auroc?.toFixed(3)??"Not available"],["ECE",bench.ece.toFixed(3)],["Brier",bench.brier.toFixed(3)],["Incorrect",`${bench.incorrect}/${bench.count}`]].map(x=><div className="panel metric" key={x[0]}><span>{x[0]}</span><strong>{x[1]}</strong><small>computed this run</small></div>)}</div>
      <div className="panel calibration"><header><div><span className="eyebrow">RELIABILITY</span><h3>Predicted vs observed risk</h3></div></header><div className="cal-bars">{bench.bins.map(b=><div key={b.range}><div><i style={{height:`${b.predicted*100}%`}}/><i className="observed" style={{height:`${b.observed*100}%`}}/></div><span>{b.range}</span><small>n={b.count}</small></div>)}</div><div className="legend"><span><i/>Predicted</span><span><i className="observed"/>Observed incorrectness</span></div></div>
      <div className="panel table-wrap"><table><thead><tr><th>Question</th><th>Reference</th><th>Outcome</th><th>Risk</th></tr></thead><tbody>{bench.records.map(r=><tr key={r.question}><td data-label="Question">{r.question}</td><td data-label="Reference">{r.reference}</td><td data-label="Outcome"><span className={r.correct?"correct":"incorrect"}>{r.correct?"Correct":"Incorrect"}</span></td><td data-label="Risk">{Math.round(r.risk*100)}</td></tr>)}</tbody></table></div>
    </div>:<div className="empty bench-empty"><FlaskConical/><h3>No benchmark claims before the run.</h3><p>Start the benchmark to compute AUROC, calibration error, and Brier score from actual records.</p></div>}
  </section>;
}

function Method() {
  return <section id="method" className="section method">
    <div className="section-heading"><div><span className="eyebrow">04 · METHODOLOGY</span><h2>Four imperfect signals. One transparent estimate.</h2></div><p>Mirage estimates observed hallucination risk. It does not independently establish truth.</p></div>
    <div className="pipeline">{["Question","Primary generation","Token uncertainty","Temperature samples","Semantic clustering","Self-verification","MirageScore"].map((x,i)=><div key={x}><span>{String(i+1).padStart(2,"0")}</span><b>{x}</b>{i<6&&<ArrowRight/>}</div>)}</div>
    <div className="method-grid"><article className="panel"><BookOpen/><h3>What Mirage measures</h3><ul><li>Disagreement across sampled outputs</li><li>Token-generation uncertainty</li><li>Model self-verification confidence</li><li>Sensitivity to prompt perturbations</li><li>Correlation with benchmark correctness</li></ul></article><article className="panel warning-panel"><ShieldAlert/><h3>What Mirage does not prove</h3><ul><li>Semantic agreement can be consistently wrong.</li><li>High confidence does not guarantee correctness.</li><li>P(True) may itself be miscalibrated.</li><li>NLI equivalence can fail.</li><li>A score cannot replace source verification.</li></ul></article></div>
  </section>;
}

export default function App() {
  const [online,setOnline]=useState(false),[latest,setLatest]=useState<Analysis|null>(null),[system,setSystem]=useState<Record<string,unknown>>({});
  useEffect(()=>{api.health().then(()=>setOnline(true)).catch(()=>setOnline(false));api.system().then(setSystem).catch(()=>{})},[]);
  const launch=()=>document.getElementById("playground")?.scrollIntoView({behavior:"smooth"});
  return <><Nav online={online}/><main><Hero launch={launch}/><section className="signal-strip">{signals.map(([Icon,title,text],i)=><article key={title}><span>0{i+1}</span><Icon/><div><h3>{title}</h3><p>{text}</p></div></article>)}</section><Playground onAnalysis={setLatest}/><Stress latest={latest}/><BenchSection/><Method/></main>
    <footer><div className="brand"><span className="brand-mark"/><b>MIRAGE</b></div><p>A research and evaluation tool—not a medical, legal, or financial decision-maker.</p><a href="https://github.com/aashita-46/Mirage-LLM"><Github size={17}/> Source</a></footer>
    <dialog id="system"><button className="close" onClick={()=>(document.getElementById("system") as HTMLDialogElement | null)?.close()}><X/></button><span className="eyebrow">SYSTEM DIAGNOSTICS</span><h2>Mirage status</h2><div className="diagnostics">{Object.entries(system).map(([k,v])=><div key={k}><span>{k.replace(/([A-Z])/g," $1")}</span><b>{typeof v==="boolean"?(v?"Available":"Unavailable"):String(v)}</b></div>)}</div><p className="micro">Secrets and local paths are never exposed.</p></dialog>
  </>;
}
