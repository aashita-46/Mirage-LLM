import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity, AlertTriangle, ArrowRight, BarChart3, Beaker, BookOpen, Boxes, BrainCircuit,
  CheckCircle2, ChevronRight, CircleGauge, Code2, Database, Download, FileJson, FlaskConical,
  Gauge, Github, Layers3, Linkedin, Menu, Play, RefreshCw, Scale, Search, Settings, ShieldAlert,
  SlidersHorizontal, Table2, Trash2, Upload, X, XCircle
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";
import { api } from "./lib/api";
import type {
  AggregateMetrics, Analysis, DatasetManifest, DatasetSummary, ExampleResult,
  ExperimentRecord, ExperimentSummary, ProviderModel
} from "./types";

type View = "overview"|"playground"|"experiments"|"datasets"|"compare"|"calibration"|"failures"|"findings"|"reports"|"methodology"|"settings";
const nav: [View,string][] = [
  ["playground","Playground"],["experiments","Experiments"],["datasets","Datasets"],
  ["compare","Compare"],["calibration","Calibration"],["failures","Failures"],
  ["findings","Findings"],["reports","Reports"],["methodology","Methodology"]
];
const starterConfig = {
  experiment_name:"Starter uncertainty-signal study",dataset_name:"mirage-starter",dataset_version:"1.0",
  provider:"cached_demo",model:"mirage/cached-research-samples",
  prompt_template:"{question}",system_prompt:"Answer concisely. If the premise is false or unknowable, say so.",
  sampling:{temperature:.7,top_p:.9,max_tokens:180,seed:42,semantic_samples:6},
  signals:{semantic_entropy:true,response_consistency:true,self_verification:true,token_uncertainty:false,
    retrieval_faithfulness:false,clustering_method:"lexical_fallback",
    weights:{semantic_entropy:.45,response_inconsistency:.30,self_verification_uncertainty:.25}},
  evaluator:{methods:["acceptable_answer","normalised_exact_match","token_f1"],numeric_tolerance:.01,llm_judge_enabled:false},
  retrieval_enabled:false,retrieval_config:{}
};

const pct=(value?:number|null,digits=1)=>value==null?"N/A":`${(value*100).toFixed(digits)}%`;
const num=(value?:number|null,digits=3)=>value==null?"N/A":value.toFixed(digits);
const titleCase=(text:string)=>text.replace(/_/g," ").replace(/\b\w/g,(x:string)=>x.toUpperCase());
const researchPillars: Array<{Icon: typeof Database; title:string; copy:string}> = [
  {Icon:Database,title:"Labelled data",copy:"Versioned JSON/JSONL datasets with traceable references."},
  {Icon:BrainCircuit,title:"Signal comparison",copy:"Evaluate disagreement, consistency, verification, and provider-native token signals."},
  {Icon:Scale,title:"Calibration",copy:"Measure ECE, Brier score, reliability, and selective prediction."},
  {Icon:ShieldAlert,title:"Failure analysis",copy:"Find confidently wrong, uncertain-correct, false-premise, and provider failures."},
];

function Navigation({view,setView,online}:{view:View;setView:(v:View)=>void;online:boolean}) {
  const [open,setOpen]=useState(false);
  return <nav className="research-nav">
    <button className="brand brand-button" onClick={()=>setView("overview")}><span className="brand-mark"/><b>MIRAGE</b><em>RESEARCH</em></button>
    <button className="menu" aria-expanded={open} onClick={()=>setOpen(!open)}>{open?<X/>:<Menu/>}</button>
    <div className={`nav-links ${open?"open":""}`}>{nav.map(([id,label])=><button key={id} className={view===id?"active":""} onClick={()=>{setView(id);setOpen(false)}}>{label}</button>)}</div>
    <button className="system-chip" onClick={()=>setView("settings")}><i className={online?"ok":""}/>{online?"Local research":"Disconnected"}<Settings size={14}/></button>
  </nav>;
}

function Metric({label,value,note,tone}:{label:string;value:string;note:string;tone?:string}) {
  return <article className={`panel metric-card ${tone??""}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>;
}

function Overview({setView}:{setView:(v:View)=>void}) {
  return <main>
    <header className="research-hero">
      <motion.div initial={{opacity:0,y:18}} animate={{opacity:1,y:0}}>
        <div className="overline"><Beaker size={14}/> LLM RELIABILITY EVALUATION</div>
        <h1>Make model uncertainty <span>measurable.</span></h1>
        <p>Mirage evaluates whether uncertainty signals, model disagreement, and retrieval faithfulness can predict factual errors in large language model outputs.</p>
        <div className="hero-actions"><button className="primary" onClick={()=>setView("experiments")}>Run an evaluation <ArrowRight size={17}/></button><button className="secondary" onClick={()=>setView("playground")}>Open playground</button><button className="text-button" onClick={()=>setView("methodology")}>View methodology</button></div>
        <div className="honesty-line"><CheckCircle2/> Uncertainty is evaluated against labelled correctness—never treated as truth.</div>
      </motion.div>
      <div className="research-visual panel">
        <header><div><span className="eyebrow">RESEARCH QUESTION</span><h3>Which signal best separates errors?</h3></div><span className="demo-label">CACHED DEMO CONFIG</span></header>
        <div className="signal-preview">
          {[["Semantic entropy",.72],["Response inconsistency",.61],["Self-verification uncertainty",.46],["Token uncertainty",null]].map(([name,value])=>
            <div key={String(name)}><span>{name}</span>{value==null?<em>Unavailable</em>:<><div className="bar"><i style={{width:`${Number(value)*100}%`}}/></div><b>{Number(value).toFixed(2)}</b></>}</div>)}
        </div>
        <p className="micro">Illustrative interface preview only. Experiment metrics appear after a reproducible local run.</p>
      </div>
    </header>
    <section className="research-pillars">
      {researchPillars.map(({Icon,title,copy})=>
        <article className="panel" key={title}><Icon/><h3>{title}</h3><p>{copy}</p></article>)}
    </section>
    <section className="positioning panel"><AlertTriangle/><div><h3>What Mirage does not claim</h3><p>Mirage does not determine factual correctness solely from model uncertainty. It measures whether selected signals correlate with labelled errors on a defined dataset.</p></div></section>
  </main>;
}

function Playground() {
  const [question,setQuestion]=useState("What is the capital of Australia?");
  const [reference,setReference]=useState("Canberra");
  const [samples,setSamples]=useState(6),[temperature,setTemperature]=useState(.7);
  const [result,setResult]=useState<Analysis|null>(null),[loading,setLoading]=useState(false),[error,setError]=useState("");
  const run=async()=>{setLoading(true);setError("");try{setResult(await api.analyse(question,reference,samples,temperature))}catch(e){setError(String(e))}finally{setLoading(false)}};
  return <Page eyebrow="INTERACTIVE PLAYGROUND" title="Inspect signals without confusing them with correctness." intro="Use a cached starter question or configure a live provider later. Provider capability flags decide which signals are available.">
    <div className="panel playground-form">
      <label>Question<textarea value={question} onChange={e=>setQuestion(e.target.value)}/></label>
      <div className="form-grid"><label>Reference answer<input value={reference} onChange={e=>setReference(e.target.value)} placeholder="Optional labelled answer"/></label><label>Semantic samples<select value={samples} onChange={e=>setSamples(+e.target.value)}>{[4,6,8,10].map(v=><option key={v}>{v}</option>)}</select></label><label>Temperature <b>{temperature}</b><input type="range" min="0" max="1.5" step=".1" value={temperature} onChange={e=>setTemperature(+e.target.value)}/></label></div>
      <div className="capability-row"><span><CheckCircle2/> Multiple cached samples</span><span className="disabled"><XCircle/> Token log-probabilities unavailable</span><span className="disabled"><XCircle/> Retrieval not configured</span></div>
      <button className="primary" disabled={loading} onClick={run}>{loading?<RefreshCw className="spin"/>:<Play/>}{loading?"Evaluating…":"Evaluate example"}</button>
    </div>
    {error&&<ErrorState message={error}/>}
    {result?.error&&<ErrorState message={result.error}/>}
    {result&&!result.error&&<div className="playground-results">
      <div className="metric-grid"><Metric label="Mirage Risk Score" value={result.score==null?"N/A":result.score.toFixed(1)} note="Experimental composite · not factual truth"/><Metric label="Semantic entropy" value={num(result.semanticEntropy)} note="Lexical fallback clustering"/><Metric label="Self-verification" value={pct(result.pTrue)} note="Cached model signal · not ground truth"/><Metric label="Correctness" value={String(result.correctness?.correct??"Not labelled")} note={String(result.correctness?.method??"No evaluator")}/></div>
      <div className="result-grid">
        <article className="panel"><span className="eyebrow">PRIMARY RESPONSE · CACHED</span><h3>{result.answer}</h3><p className="trace">Matched dataset item: {result.matchedDatasetQuestion}</p></article>
        <article className="panel unavailable"><XCircle/><div><h3>Token uncertainty unavailable</h3><p>{result.tokenSignal?.reason}</p></div></article>
        <article className="panel cluster-view"><span className="eyebrow">SEMANTIC GROUPS</span>{result.clusters.map(c=><div key={c.id}><b>{c.label}</b><span>{pct(c.probability,0)}</span><p>{c.representativeAnswer}</p><small>{c.evaluatorMethod}</small></div>)}</article>
        <article className="panel"><span className="eyebrow">RAW TRACE</span><pre>{JSON.stringify({metadata:result.metadata,capabilities:result.capabilities,breakdown:result.breakdown},null,2)}</pre></article>
      </div>
    </div>}
  </Page>;
}

function Datasets() {
  const [datasets,setDatasets]=useState<DatasetSummary[]>([]),[detail,setDetail]=useState<DatasetManifest|null>(null);
  const [domain,setDomain]=useState("all"),[difficulty,setDifficulty]=useState("all");
  const [validation,setValidation]=useState<Record<string,unknown>|null>(null),[error,setError]=useState("");
  const [uploadFile,setUploadFile]=useState<{name:string;content:string}|null>(null),[saved,setSaved]=useState("");
  useEffect(()=>{api.datasets().then(x=>{setDatasets(x.datasets);if(x.datasets[0])api.dataset(x.datasets[0].name).then(setDetail)}).catch(e=>setError(String(e)))},[]);
  const upload=async(e:React.ChangeEvent<HTMLInputElement>)=>{const file=e.target.files?.[0];if(!file)return;try{const content=await file.text();setUploadFile({name:file.name,content});setValidation(await api.validateDataset(file.name,content));setSaved("")}catch(err){setError(String(err))}};
  const saveUpload=async()=>{if(!uploadFile)return;try{const result=await api.saveDataset(uploadFile.name,uploadFile.content);setSaved(`${result.name} v${result.version} saved locally (${result.size} examples).`);const list=await api.datasets();setDatasets(list.datasets)}catch(err){setError(String(err))}};
  const filtered=detail?.examples.filter(x=>(domain==="all"||x.domain===domain)&&(difficulty==="all"||x.difficulty===difficulty))??[];
  return <Page eyebrow="VERSIONED DATASETS" title="Define what correctness means before measuring uncertainty." intro="Mirage accepts JSON and JSONL. Every example must be validated before it can enter an experiment.">
    {error&&<ErrorState message={error}/>}
    <div className="dataset-grid"><aside className="panel dataset-list">{datasets.map(d=><button key={d.name} onClick={()=>api.dataset(d.name).then(setDetail)}><Database/><span><b>{d.name}</b><small>v{d.version} · {d.size} examples</small></span><ChevronRight/></button>)}</aside>
      <section className="panel dataset-detail">{detail&&<><header><div><span className="eyebrow">DEMONSTRATION DATASET</span><h3>{detail.name} <small>v{detail.version}</small></h3></div><span className="demo-label">NOT A DEFINITIVE BENCHMARK</span></header><p>{detail.description}</p><div className="dataset-meta"><span>{detail.examples.length} examples</span><span>{new Set(detail.examples.map(x=>x.domain)).size} domains</span><span>{detail.license}</span></div><div className="filter-row"><label>Domain<select value={domain} onChange={e=>setDomain(e.target.value)}><option value="all">All domains</option>{[...new Set(detail.examples.map(x=>x.domain))].map(x=><option key={x}>{x}</option>)}</select></label><label>Difficulty<select value={difficulty} onChange={e=>setDifficulty(e.target.value)}><option value="all">All levels</option><option>easy</option><option>medium</option><option>hard</option></select></label></div><div className="example-table">{filtered.slice(0,8).map(x=><article key={x.id}><span>{x.id}</span><div><b>{x.question}</b><small>{x.reference_answer??"Explicitly unanswerable"} · {x.source}</small></div><em>{x.domain} / {x.difficulty}</em></article>)}</div></>}</section>
    </div>
    <section className="panel upload-panel"><Upload/><div><h3>Validate and save a dataset</h3><p>Files remain local. Malformed rows are reported before a versioned manifest is written.</p>{saved&&<small className="correct">{saved}</small>}</div><label className="secondary file-button">Choose JSON or JSONL<input type="file" accept=".json,.jsonl" onChange={upload}/></label>{validation?.valid===true&&<button className="primary" onClick={saveUpload}>Save validated dataset</button>}</section>
    {validation&&<pre className="panel validation-result">{JSON.stringify(validation,null,2)}</pre>}
  </Page>;
}

function Experiments({onSelected}:{onSelected:(r:ExperimentRecord)=>void}) {
  const [items,setItems]=useState<ExperimentSummary[]>([]),[loading,setLoading]=useState(false),[error,setError]=useState("");
  const [name,setName]=useState("Starter uncertainty-signal study"),[sampleCount,setSampleCount]=useState(6);
  const [datasets,setDatasets]=useState<DatasetSummary[]>([]),[datasetName,setDatasetName]=useState("mirage-starter");
  const [semanticEntropy,setSemanticEntropy]=useState(true),[responseConsistency,setResponseConsistency]=useState(true);
  const [selfVerification,setSelfVerification]=useState(true);
  const [models,setModels]=useState<ProviderModel[]>([]),[selectedModel,setSelectedModel]=useState("cached_demo|mirage/cached-research-samples");
  const [semanticMethod,setSemanticMethod]=useState("lexical_fallback");
  const refresh=()=>api.experiments().then(x=>setItems(x.experiments)).catch(e=>setError(String(e)));
  useEffect(()=>{refresh();api.datasets().then(x=>setDatasets(x.datasets)).catch(()=>{});api.models().then(x=>setModels(x.models)).catch(()=>{})},[]);
  const run=async()=>{
    setLoading(true);setError("");
    const dataset=datasets.find(item=>item.name===datasetName);
    const [provider,model]=selectedModel.split("|",2);
    try{
      const record=await api.runExperiment({
        ...starterConfig,experiment_name:name,dataset_name:datasetName,provider,model,
        dataset_version:dataset?.version??"1.0",
        sampling:{...starterConfig.sampling,semantic_samples:sampleCount},
        signals:{...starterConfig.signals,semantic_entropy:semanticEntropy,clustering_method:semanticMethod,
          response_consistency:responseConsistency,self_verification:selfVerification},
      });
      onSelected(record);refresh();
    }catch(e){setError(String(e))}finally{setLoading(false)}
  };
  return <Page eyebrow="EXPERIMENT RUNNER" title="Run reproducible evaluations—not dashboard theatre." intro="Each run saves raw cached responses, evaluator outputs, signal configuration, schema version, and aggregate results to SQLite.">
    <div className="runner-layout"><section className="panel runner-config"><h3>Experiment configuration</h3><label>Name<input value={name} onChange={e=>setName(e.target.value)}/></label><label>Dataset<select value={datasetName} onChange={e=>setDatasetName(e.target.value)}>{datasets.map(d=><option value={d.name} key={d.name}>{d.name} v{d.version} · {d.size} examples</option>)}</select></label><label>Provider / model<select value={selectedModel} onChange={e=>setSelectedModel(e.target.value)}>{models.filter(x=>x.available!==false).map(x=><option value={`${x.provider}|${x.model}`} key={`${x.provider}|${x.model}`}>{x.mode==="cached_demo"?"Cached demo":`${x.provider} · ${x.model}`}</option>)}</select></label><label>Semantic method<select value={semanticMethod} onChange={e=>setSemanticMethod(e.target.value)}><option value="lexical_fallback">Lexical Jaccard fallback</option><option value="embedding">Embedding cosine · all-minilm</option></select></label><label>Semantic samples<select value={sampleCount} onChange={e=>setSampleCount(+e.target.value)}>{[4,6,8,10].map(x=><option key={x}>{x}</option>)}</select></label><fieldset><legend>Signals</legend><label><input type="checkbox" checked={semanticEntropy} onChange={e=>setSemanticEntropy(e.target.checked)}/> Semantic entropy</label><label><input type="checkbox" checked={responseConsistency} onChange={e=>setResponseConsistency(e.target.checked)}/> Response consistency</label><label><input type="checkbox" checked={selfVerification} onChange={e=>setSelfVerification(e.target.checked)}/> Self-verification</label><label className="disabled"><input type="checkbox" disabled/> Token uncertainty · capability dependent</label><label className="disabled"><input type="checkbox" disabled/> Retrieval faithfulness · not configured</label></fieldset><button className="primary" disabled={loading||(!semanticEntropy&&!responseConsistency&&!selfVerification)} onClick={run}>{loading?<RefreshCw className="spin"/>:<FlaskConical/>}{loading?"Running labelled examples…":"Run evaluation"}</button>{!semanticEntropy&&!responseConsistency&&!selfVerification&&<small className="warning-copy">Enable at least one available signal.</small>}</section>
      <section className="panel run-explainer"><h3>What this run will do</h3>{["Load and validate the versioned dataset","Read cached provider outputs without inventing logits","Cluster sampled responses with labelled lexical fallback","Evaluate correctness against references and aliases","Compute calibration and selective prediction metrics","Persist raw outputs, metrics, and failure categories"].map((x,i)=><div key={x}><span>{i+1}</span><p>{x}</p></div>)}<p className="warning-copy"><AlertTriangle/> Cached results demonstrate the complete pipeline but are not a newly executed LLM benchmark.</p></section></div>
    {error&&<ErrorState message={error}/>}
    <h3 className="subheading">Saved experiments</h3>
    {items.length?<div className="experiment-list">{items.map(item=><button className="panel" key={item.experiment_id} onClick={()=>api.experiment(item.experiment_id).then(onSelected)}><span className={`run-state ${item.state}`}>{item.state}</span><div><b>{item.experiment_name}</b><small>{item.experiment_id} · schema {item.schema_version}</small></div><strong>{pct(item.aggregates?.accuracy)}</strong><span>accuracy</span><ChevronRight/></button>)}</div>:<Empty title="No experiments saved yet." copy="Run the starter study to create a traceable local experiment."/>}
  </Page>;
}

function ExperimentDetail({record,onDelete}:{record:ExperimentRecord;onDelete:()=>void}) {
  const metrics=record.aggregates;
  return <Page eyebrow={`EXPERIMENT · ${record.state.toUpperCase()}`} title={record.experiment_name} intro={`${record.experiment_id} · schema ${record.schema_version} · ${String(record.dataset.name)} ${String(record.dataset.version)}`}>
    <div className="detail-actions"><button className="secondary danger" onClick={onDelete}><Trash2/> Delete experiment</button></div>
    {metrics&&<><MetricGrid metrics={metrics}/><Warnings warnings={metrics.warnings}/>
    <div className="chart-grid"><Reliability metrics={metrics}/><RiskCoverage metrics={metrics}/></div>
    <section className="panel"><header><div><span className="eyebrow">SIGNAL COMPARISON</span><h3>No signal is assumed best.</h3></div></header><SignalTable metrics={metrics}/></section>
    <section className="panel"><header><div><span className="eyebrow">TRACEABLE RESULTS</span><h3>Per-example outputs</h3></div></header><ResultTable results={record.results}/></section></>}
  </Page>;
}

function Compare({experiments}:{experiments:ExperimentSummary[]}) {
  return <Page eyebrow="EXPERIMENT COMPARISON" title="Compare configurations on equal footing." intro="Saved experiments can be compared across models, prompts, decoding parameters, and signal configurations.">
    {experiments.length?<><div className="compare-grid">{experiments.map(x=><article className="panel" key={x.experiment_id}><span className="demo-label">{x.state}</span><h3>{x.experiment_name}</h3><small>{String(x.model.model)} · {String(x.dataset.name)}</small><div className="compare-metrics"><span>Accuracy <b>{pct(x.aggregates?.accuracy)}</b></span><span>AUROC <b>{num(x.aggregates?.auroc)}</b></span><span>AUPRC <b>{num(x.aggregates?.auprc)}</b></span><span>ECE <b>{num(x.aggregates?.ece)}</b></span></div></article>)}</div><p className="panel warning-copy"><AlertTriangle/> Identical deterministic IDs indicate identical evaluation configuration; reruns replace rather than duplicate the record.</p></>:<Empty title="No comparable experiments." copy="Run at least one experiment first. Add a second configuration to compare changes."/>}
  </Page>;
}

function CalibrationPage({record}:{record:ExperimentRecord|null}) {
  return <Page eyebrow="CALIBRATION & SELECTIVE PREDICTION" title="Route risk to review instead of trusting a gauge." intro="Calibration is only computed for bounded risk estimates with correctness labels. Small samples receive an explicit warning.">
    {record?.aggregates?<><Warnings warnings={record.aggregates.warnings}/><div className="chart-grid"><Reliability metrics={record.aggregates}/><RiskCoverage metrics={record.aggregates}/></div><ThresholdPanel metrics={record.aggregates}/></>:<Empty title="Run or open an experiment." copy="Calibration and risk-coverage curves require labelled per-example results."/>}
  </Page>;
}

function ThresholdPanel({metrics}:{metrics:AggregateMetrics}) {
  const [target,setTarget]=useState(.9);
  const eligible=metrics.risk_coverage.filter(x=>x.selective_accuracy>=target);
  const point=eligible.length?eligible.reduce((best,row)=>row.coverage>best.coverage?row:best):undefined;
  return <section className="panel threshold-panel"><header><div><span className="eyebrow">HUMAN REVIEW POLICY</span><h3>Target selective accuracy</h3></div><strong>{pct(target,0)}</strong></header><input type="range" min=".5" max="1" step=".01" value={target} onChange={e=>setTarget(+e.target.value)}/>{point?<div className="metric-grid"><Metric label="Automation coverage" value={pct(point.coverage)} note="Examples accepted"/><Metric label="Review rate" value={pct(point.review_rate)} note="Examples escalated"/><Metric label="Selective accuracy" value={pct(point.selective_accuracy)} note="Accuracy among accepted"/><Metric label="Remaining errors" value={String(point.remaining_errors)} note="Within accepted examples"/></div>:<p>No valid threshold is available.</p>}</section>;
}

function Failures({record,onUpdated}:{record:ExperimentRecord|null;onUpdated:(record:ExperimentRecord)=>void}) {
  const [filter,setFilter]=useState("all");
  const failures=record?.results.filter(x=>x.failure_types.length)??[];
  const shown=failures.filter(x=>filter==="all"||x.failure_types.includes(filter));
  const types=[...new Set(failures.flatMap(x=>x.failure_types))];
  return <Page eyebrow="FAILURE ANALYSIS" title="Study the cases aggregate metrics hide." intro="Every failure links back to raw outputs, reference labels, evaluator rationale, signal values, and configuration.">
    {!record?<Empty title="No experiment selected." copy="Run or open an experiment to inspect its failure slices."/>:<><div className="filter-row"><label>Failure slice<select value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">All failures</option>{types.map(x=><option key={x} value={x}>{titleCase(x)}</option>)}</select></label><span>{shown.length} of {record.results.length} examples</span></div><div className="failure-grid">{shown.map(x=><FailureCard key={x.example_id} experimentId={record.experiment_id} result={x} onUpdated={onUpdated}/>)}</div></>}
  </Page>;
}
function FailureCard({experimentId,result,onUpdated}:{experimentId:string;result:ExampleResult;onUpdated:(record:ExperimentRecord)=>void}) {
  const [raw,setRaw]=useState(false),[saving,setSaving]=useState(false),[error,setError]=useState("");
  const override=async(label:boolean)=>{setSaving(true);setError("");try{onUpdated(await api.override(experimentId,result.example_id,label,"Reviewed in Mirage failure explorer."))}catch(e){setError(String(e))}finally{setSaving(false)}};
  return <article className="panel failure-card"><header><div>{result.failure_types.map(x=><span className="failure-tag" key={x}>{titleCase(x)}</span>)}</div><b>{result.predicted_risk==null?"N/A":pct(result.predicted_risk)}</b></header><h3>{result.question}</h3><dl><dt>Reference</dt><dd>{result.reference_answer??"Explicitly unanswerable"}</dd><dt>Response</dt><dd>{result.raw_generation.response}</dd><dt>Evaluator</dt><dd>{result.correctness.reason}</dd><dt>Effective label</dt><dd>{String(result.correctness.human_label??result.correctness.correct)}{result.correctness.human_label!=null?" · human reviewed":""}</dd></dl><div className="override-actions"><button className="secondary" disabled={saving} onClick={()=>override(true)}>Mark correct</button><button className="secondary danger" disabled={saving} onClick={()=>override(false)}>Mark incorrect</button><button className="secondary" onClick={()=>setRaw(!raw)}>{raw?"Hide":"Show"} raw JSON</button></div>{error&&<small className="incorrect">{error}</small>}{raw&&<pre>{JSON.stringify(result,null,2)}</pre>}</article>;
}

function Reports({record}:{record:ExperimentRecord|null}) {
  return <Page eyebrow="REPORTS & EXPORT" title="Carry the configuration with the conclusion." intro="Exports include experiment configuration, raw outputs, metrics, failures, schema version, and limitations.">
    {record?<div className="report-card panel"><FileJson/><div><h3>{record.experiment_name}</h3><p>{record.experiment_id} · {record.results.length} examples · {record.state}</p></div><div className="export-actions">{["json","csv","markdown","html"].map(format=><a className="secondary" href={api.exportUrl(record.experiment_id,format)} key={format} download><Download/> {format.toUpperCase()}</a>)}</div></div>:<Empty title="No report is selected." copy="Run or open an experiment before exporting."/>}
  </Page>;
}

function Findings() {
  const [data,setData]=useState<Record<string,unknown>|null>(null),[error,setError]=useState("");
  const [group,setGroup]=useState("live-smoke-v1");
  useEffect(()=>{setData(null);api.findings(group).then(setData).catch(e=>setError(String(e)))},[group]);
  const available=data?.available===true;
  const executive=(data?.executive_findings??{}) as Record<string,unknown>;
  const experiments=(data?.experiments??[]) as Array<Record<string,unknown>>;
  return <Page eyebrow="RESEARCH FINDINGS" title="Answer the experiment—not the dashboard." intro="This page displays only measurements derived from stored live experiments, with preliminary smoke runs separated from the official research group.">
    <div className="filter-row"><label>Experiment group<select value={group} onChange={e=>setGroup(e.target.value)}><option value="live-smoke-v1">Live smoke v1 · preliminary</option><option value="research-v1">Research v1 · official</option></select></label></div>
    {group!=="research-v1"&&<div className="warning-copy panel"><AlertTriangle/> Preliminary pipeline validation only; this group cannot answer the primary research question.</div>}
    {error&&<ErrorState message={error}/>}
    {!data?<Empty title="Loading measured findings." copy="Reading locally stored research experiments."/>:!available?<Empty title="No official findings yet." copy={String(data.reason)}/>:<>
      <section className="panel findings-summary"><span className="eyebrow">EXECUTIVE FINDINGS · MEASURED</span><pre>{JSON.stringify(executive,null,2)}</pre></section>
      <div className="compare-grid">{experiments.map(item=><article className="panel" key={String(item.experiment_id)}><span className="demo-label">LIVE EXPERIMENT</span><h3>{String(item.model)}</h3><small>{String(item.experiment_id)}</small><pre>{JSON.stringify(item.metrics,null,2)}</pre></article>)}</div>
    </>}
  </Page>;
}

function Methodology() {
  const topics=[
    ["Token surprisal","s(x) = −log p(x)","Provider-native log-probabilities only. Mirage disables this signal when unavailable."],
    ["Semantic entropy","H = −Σ p(c) log p(c)","Entropy over meaning clusters. The local fallback uses lexical Jaccard and is not neural NLI."],
    ["Embedding equivalence","cos(a,b) ≥ τ","Local all-minilm cosine clustering stores the model, threshold, membership, and similarity matrix."],
    ["Expected Calibration Error","Σ (nᵦ/N) |riskᵦ − errorᵦ|","Compares bounded predicted error risk with observed incorrectness."],
    ["Brier score","(1/N) Σ (riskᵢ − errorᵢ)²","A proper scoring rule for probabilistic error predictions."],
    ["AUROC / AUPRC","ranking discrimination","Measures separation of incorrect and correct answers; AUPRC is important with rare errors."],
    ["Selective prediction","accept low-risk outputs; review the rest","Risk-coverage and selective accuracy quantify the human-review trade-off."],
  ];
  return <Page eyebrow="METHODOLOGY" title="Transparent metrics, explicit assumptions." intro="Mirage evaluates correlations between signals and labelled errors. It does not infer truth from confidence alone.">
    <div className="method-cards">{topics.map(([title,formula,copy])=><article className="panel" key={title}><h3>{title}</h3><code>{formula}</code><p>{copy}</p></article>)}</div>
    <section className="panel limitations"><AlertTriangle/><div><h3>Required limitation</h3><p>Mirage does not determine whether an answer is factually correct solely from model uncertainty. It evaluates whether selected uncertainty and consistency signals correlate with labelled errors on a defined dataset. Results depend on the dataset, model, evaluator, prompt, and sampling configuration.</p></div></section>
    <section className="panel references"><h3>Threats to validity</h3><p>Dataset size and composition, reference quality, model-selection bias, local hardware, sampling sensitivity, clustering errors, judge disagreement, self-verification unreliability, class imbalance, bootstrap assumptions, prompt dependence, and provider drift can all change conclusions.</p></section>
    <section className="panel references"><h3>Selected references</h3><ul><li>Kuhn et al., “Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation” (2023).</li><li>Guo et al., “On Calibration of Modern Neural Networks” (2017).</li><li>Geifman and El-Yaniv, “Selective Classification for Deep Neural Networks” (2017).</li><li>Lin et al., “Teaching Models to Express Their Uncertainty in Words” (2022).</li></ul></section>
  </Page>;
}

function SettingsPage({system}:{system:Record<string,unknown>}) {
  return <Page eyebrow="SYSTEM & REPRODUCIBILITY" title="Know exactly what is—and is not—running." intro="The UI reacts to provider capability flags. Unsupported controls remain disabled rather than simulated.">
    <div className="settings-grid">{Object.entries(system).map(([key,value])=><article className="panel" key={key}><span>{titleCase(key)}</span><strong>{typeof value==="object"?JSON.stringify(value):String(value)}</strong></article>)}</div>
    <section className="panel command-card"><Code2/><div><h3>Local commands</h3><pre>python -m uvicorn api.index:app --reload --port 8000{"\n"}npm run dev{"\n"}python -m pytest -q{"\n"}npm test{"\n"}npm run build</pre></div></section>
  </Page>;
}

function Page({eyebrow,title,intro,children}:{eyebrow:string;title:string;intro:string;children:React.ReactNode}) {
  return <main className="research-page"><header className="page-heading"><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{intro}</p></header>{children}</main>;
}
function Empty({title,copy}:{title:string;copy:string}){return <div className="empty"><FlaskConical/><h3>{title}</h3><p>{copy}</p></div>}
function ErrorState({message}:{message:string}){return <div className="error-state"><XCircle/><div><b>Evaluation stage failed</b><p>{message}</p></div></div>}
function Warnings({warnings}:{warnings:string[]}){return <>{warnings.map(x=><div className="warning-copy panel" key={x}><AlertTriangle/>{x}</div>)}</>}
function MetricGrid({metrics}:{metrics:AggregateMetrics}){
  const reason=(name:string,fallback:string)=>metrics.metric_status?.[name]?.available===false?metrics.metric_status[name].reason??"Unavailable":fallback;
  return <div className="metric-grid"><Metric label="Accuracy" value={pct(metrics.accuracy)} note={reason("accuracy",`${metrics.labelled_examples} labelled examples`)}/><Metric label="AUROC" value={num(metrics.auroc)} note={reason("auroc","Higher means better error ranking; it does not establish truth")}/><Metric label="AUPRC" value={num(metrics.auprc)} note={reason("auprc","Incorrectness is the positive class")}/><Metric label="ECE" value={num(metrics.ece)} note={reason("calibration","Lower is better; dataset-conditional calibration")}/><Metric label="Brier" value={num(metrics.brier)} note={reason("calibration","Lower is better; bounded error risk")}/><Metric label="p95 latency" value={metrics.p95_latency_ms==null?"N/A":`${metrics.p95_latency_ms.toFixed(0)} ms`} note="Provider-recorded latency"/></div>
}
function Reliability({metrics}:{metrics:AggregateMetrics}){return <article className="panel chart-card"><span className="eyebrow">RELIABILITY DIAGRAM</span><h3>Predicted risk vs observed error</h3><ResponsiveContainer width="100%" height={260}><BarChart data={metrics.reliability_bins}><CartesianGrid stroke="#242a36"/><XAxis dataKey="predicted" tickFormatter={x=>x.toFixed(1)}/><YAxis domain={[0,1]}/><Tooltip/><Legend/><Bar dataKey="predicted" fill="#7770ee"/><Bar dataKey="observed" fill="#61d7cd"/></BarChart></ResponsiveContainer></article>}
function RiskCoverage({metrics}:{metrics:AggregateMetrics}){return <article className="panel chart-card"><span className="eyebrow">RISK–COVERAGE</span><h3>Accuracy after escalating risky outputs</h3><ResponsiveContainer width="100%" height={260}><LineChart data={metrics.risk_coverage}><CartesianGrid stroke="#242a36"/><XAxis dataKey="coverage" tickFormatter={x=>`${Math.round(x*100)}%`}/><YAxis domain={[0,1]} tickFormatter={x=>`${Math.round(x*100)}%`}/><Tooltip/><Line dataKey="selective_accuracy" stroke="#61d7cd" strokeWidth={2} dot={false}/><Line dataKey="error_rate" stroke="#ff7187" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></article>}
function SignalTable({metrics}:{metrics:AggregateMetrics}){return <div className="responsive-table"><table><thead><tr><th>Signal</th><th>Coverage</th><th>AUROC</th><th>AUPRC</th><th>ECE</th><th>Brier</th></tr></thead><tbody>{metrics.signal_comparison.map(x=><tr key={x.signal}><td>{titleCase(x.signal)}</td><td>{pct(x.coverage)}</td><td>{num(x.auroc)}</td><td>{num(x.auprc)}</td><td>{num(x.ece)}</td><td>{num(x.brier)}</td></tr>)}</tbody></table></div>}
function ResultTable({results}:{results:ExampleResult[]}){return <div className="responsive-table"><table><thead><tr><th>Example</th><th>Domain</th><th>Correct</th><th>Risk</th><th>Signals</th><th>Failure</th></tr></thead><tbody>{results.map(x=><tr key={x.example_id}><td><b>{x.example_id}</b><small>{x.question}</small></td><td>{x.domain}</td><td>{String(x.correctness.human_label??x.correctness.correct)}</td><td>{pct(x.predicted_risk)}</td><td>{Object.entries(x.signals).filter(([,v])=>typeof v==="number").slice(0,2).map(([k,v])=><small key={k}>{titleCase(k)} {Number(v).toFixed(2)}</small>)}</td><td>{x.failure_types.map(titleCase).join(", ")||"—"}</td></tr>)}</tbody></table></div>}

function Builders() {
  const builders = [
    {
      name:"Ninad Naik",role:"Builder · Engineering & Product",
      image:"/builders/ninad-naik.jpg",
      linkedin:"https://www.linkedin.com/in/ninad-naik-274883262",
      profileLabel:"LinkedIn",
      github:"https://github.com/ninadnaik03",
    },
    {
      name:"Aashita Jolly",role:"Builder · Research & Experience",
      image:"/builders/aashita-jolly.jpg",
      linkedin:"https://www.linkedin.com/in/aashita-jolly",
      profileLabel:"LinkedIn",
      github:"https://github.com/aashita-46",
    },
    {
      name:"Codex, apparently",role:"Contributor · Did the typing, still gets no equity",
      image:"/builders/codex-contributor.png",
      linkedin:"https://openai.com/codex/",
      profileLabel:"OpenAI",
      github:"https://github.com/openai",
    },
  ];
  return <section className="builders-section" aria-labelledby="builders-title">
    <div className="builders-intro"><span className="eyebrow">THE PEOPLE BEHIND MIRAGE</span><h2 id="builders-title">Built with curiosity.<br/><em>Measured with care.</em></h2><p>Mirage is shaped by a shared belief that model reliability should be observable, reproducible, and honest.</p></div>
    <div className="builder-grid">{builders.map((builder,index)=><article className="builder-card" key={builder.name}>
      <div className="builder-number">0{index+1}</div>
      <div className="builder-photo"><img src={builder.image} alt={`${builder.name}, Mirage contributor`} loading="lazy"/></div>
      <div className="builder-copy"><span>{builder.role}</span><h3>{builder.name}</h3><div className="builder-links"><a href={builder.linkedin} target="_blank" rel="noreferrer" aria-label={`${builder.name} on ${builder.profileLabel}`}><Linkedin/> {builder.profileLabel}</a><a href={builder.github} target="_blank" rel="noreferrer" aria-label={`${builder.name} on GitHub`}><Github/> GitHub</a></div></div>
    </article>)}</div>
  </section>;
}

export default function App(){
  const [view,setView]=useState<View>("overview"),[online,setOnline]=useState(false);
  const [system,setSystem]=useState<Record<string,unknown>>({}),[experiments,setExperiments]=useState<ExperimentSummary[]>([]);
  const [selected,setSelected]=useState<ExperimentRecord|null>(null);
  const refresh=()=>api.experiments().then(x=>setExperiments(x.experiments)).catch(()=>{});
  useEffect(()=>{api.health().then(()=>setOnline(true)).catch(()=>setOnline(false));api.system().then(setSystem).catch(()=>{});refresh()},[]);
  const removeSelected=async()=>{if(!selected)return;const result=await api.deleteExperiment(selected.experiment_id);if(result.deleted){setSelected(null);setView("experiments");refresh()}};
  let content:React.ReactNode;
  if(view==="overview")content=<Overview setView={setView}/>;
  else if(view==="playground")content=<Playground/>;
  else if(view==="datasets")content=<Datasets/>;
  else if(view==="experiments")content=selected?<ExperimentDetail record={selected} onDelete={removeSelected}/>:<Experiments onSelected={record=>{setSelected(record);refresh()}}/>;
  else if(view==="compare")content=<Compare experiments={experiments}/>;
  else if(view==="calibration")content=<CalibrationPage record={selected}/>;
  else if(view==="failures")content=<Failures record={selected} onUpdated={record=>{setSelected(record);refresh()}}/>;
  else if(view==="findings")content=<Findings/>;
  else if(view==="reports")content=<Reports record={selected}/>;
  else if(view==="methodology")content=<Methodology/>;
  else content=<SettingsPage system={system}/>;
  return <><Navigation view={view} setView={v=>{if(v==="experiments")setSelected(null);setView(v)}} online={online}/>{content}<Builders/><footer><div className="brand"><span className="brand-mark"/><b>MIRAGE</b></div><p>Open evaluation infrastructure for studying LLM reliability—not a truth oracle.</p><a href="https://github.com/aashita-46/Mirage-LLM"><Github/> Source</a></footer></>;
}
