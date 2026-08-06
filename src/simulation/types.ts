export type SafetyMode = "advisory" | "interlock";
export type Severity = "info" | "advisory" | "warning" | "critical";
export type Phase = "taxi-out" | "holding-short" | "take-off-roll" | "climb" | "approach" | "short-final" | "landing-roll" | "taxi-in" | "go-around" | "holding";

export interface Point { x:number; y:number }
export interface Aircraft {
  id:string; callSign:string; type:string; kind:"arrival"|"departure"; position:Point; previous:Point;
  altitude:number; targetAltitude:number; heading:number; targetHeading:number; speed:number; phase:Phase;
  runway:string; route?:Point[]; routeIndex?:number; clearance?:string; conflict?:boolean;
}
export interface Vehicle { id:string; label:string; position:Point; route:Point[]; routeIndex:number; speed:number; authorised:boolean; stopped:boolean; conflict?:boolean }
export interface Weather { visibility:number; windSpeed:number; crosswind:number; condition:string }
export interface LogEvent { id:string; time:number; type:"system"|"atc"|"pilot"|"agent"|"decision"|"incident"; message:string; severity?:Severity }
export interface Observation {
  id:string; agent:string; time:number; severity:Severity; entities:string[]; category:string; observation:string;
  evidence:string[]; consequence:string; timeToConflict:number; confidence:number; recommendedAction:string;
}
export interface Recommendation {
  id:string; observationIds:string[]; title:string; explanation:string; severity:Severity; action:"go-around"|"cancel-takeoff"|"stop-vehicle"|"correct-altitude"|"hold";
  targetId:string; status:"active"|"accepted"|"rejected"|"modified"|"blocked"; createdAt:number; responseTime?:number;
}
export interface Scenario { id:string; name:string; short:string; duration:number; seed:number; aircraft:Aircraft[]; vehicles:Vehicle[]; injections:Injection[]; initialWeather:Weather }
export interface Injection { at:number; id:string; kind:"occupied-landing"|"crossing-takeoff"|"wrong-readback"|"weather"|"callsign"; fired?:boolean }
export interface Score { safety:number; awareness:number; communication:number; response:number; decisions:number }
export interface SimState {
  status:"idle"|"running"|"paused"|"ended"; time:number; speed:number; mode:SafetyMode; scenarioId:string;
  aircraft:Aircraft[]; vehicles:Vehicle[]; weather:Weather; runwayOccupiedBy?:string; logs:LogEvent[];
  observations:Observation[]; recommendations:Recommendation[]; incidents:string[]; prevented:number; interlocks:number; maxWorkload:number;
}
