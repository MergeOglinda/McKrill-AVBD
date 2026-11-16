// Carry only history belonging to unchanged body/field slots into a replacement pass.
import { SDF_MAX_POINTS } from './sdf-layout.ts';
/** sdfHistoryWGSL for a pass whose pair results hold `maxPoints` points. */
export const sdfHistorySource = (maxPoints: number = SDF_MAX_POINTS) => /* wgsl */ `
struct Point { x: vec4f, normal: vec4f, anchorA: vec4f, anchorB: vec4f, flags: vec4u, ellipsoidA: vec4f, ellipsoidB: vec4f }
struct Result { identity: vec4u, stats: vec4u, points: array<Point,${maxPoints}> }
@group(0) @binding(0) var<storage,read> source: array<Result>;
@group(0) @binding(1) var<storage,read_write> destination: array<Result>;
@group(0) @binding(2) var<storage,read> retained: array<u32>;
@compute @workgroup_size(64) fn copyHistory(@builtin(global_invocation_id) gid:vec3u) {
  let i=gid.x; if(i>=arrayLength(&destination)) { return; }
  destination[i].identity=vec4u(0xffffffffu,0xffffffffu,0xffffffffu,0u);
  if(i>=arrayLength(&source)) { return; }
  let p=source[i].identity;
  if(p.x>=arrayLength(&retained)||p.y>=arrayLength(&retained)) { return; }
  if(retained[p.x]!=0u&&retained[p.y]!=0u) { destination[i]=source[i]; }
}
`;
export const sdfHistoryWGSL = sdfHistorySource();
