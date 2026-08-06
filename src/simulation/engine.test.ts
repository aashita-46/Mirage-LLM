import { describe,expect,test } from "vitest";
import { initialState, resolve, tick } from "./engine";
import type { SimState } from "./types";

const advance=(state:ReturnType<typeof initialState>,seconds:number)=>{let s:SimState={...state,status:"running"};for(let i=0;i<seconds*4;i++)s=tick(s,.25);return s};
describe("deterministic ATC engine",()=>{
 test("replays the same seed identically",()=>{const a=advance(initialState("runway"),20),b=advance(initialState("runway"),20);expect(a.aircraft.map(x=>x.position)).toEqual(b.aircraft.map(x=>x.position));expect(a.observations.map(x=>x.category)).toEqual(b.observations.map(x=>x.category));});
 test("detects occupied runway from entity and clearance state",()=>{const s=advance(initialState("runway"),20);expect(s.observations.some(o=>o.category==="runway-occupancy")).toBe(true);expect(s.recommendations[0].action).toBe("go-around");});
 test("accepting go-around changes aircraft dynamics",()=>{let s=advance(initialState("runway"),20);const r=s.recommendations.find(x=>x.action==="go-around")!;s=resolve(s,r.id,"accepted");expect(s.aircraft.find(x=>x.id===r.targetId)?.phase).toBe("go-around");expect(s.prevented).toBe(1);});
 test("advisory does not automatically apply actions",()=>{const s=advance(initialState("runway","advisory"),20);expect(s.aircraft[0].phase).not.toBe("go-around");expect(s.recommendations[0].status).toBe("active");});
 test("interlock blocks eligible critical clearance",()=>{const s=advance(initialState("runway","interlock"),20);expect(s.recommendations.some(r=>r.status==="blocked")).toBe(true);expect(s.interlocks).toBe(1);});
 test("detects structured altitude readback mismatch",()=>{const s=advance(initialState("readback"),14);expect(s.observations.some(o=>o.category==="readback-mismatch")).toBe(true);});
 test("weather thresholds generate operational warning",()=>{const s=advance(initialState("weather"),17);expect(s.weather.visibility).toBe(1.2);expect(s.observations.some(o=>o.category==="unsafe-weather")).toBe(true);});
 test("takeoff and vehicle paths produce runway conflict",()=>{const s=advance(initialState("crossing"),16);expect(s.observations.some(o=>o.category==="runway-crossing")).toBe(true);});
});
