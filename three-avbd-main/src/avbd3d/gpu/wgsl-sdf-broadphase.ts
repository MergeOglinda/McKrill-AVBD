// GPU-generated SDF candidates. Each regular body occupies one hashed centre cell;
// exact cell checks prevent hash aliases producing duplicate pairs. Oversized bodies
// have an unbounded (body-count-limited) list, rather than the old 64-body ceiling.
import { PRELUDE_3D } from './layout.ts';
import { IGNORE_PAIRS_WGSL } from './wgsl-collision.ts';
import { SDF_MAX_POINTS } from './sdf-layout.ts';
/** sdfBroadphaseWGSL for a pass whose pair results hold `maxPoints` points. */
export const sdfBroadphaseSource = (maxPoints: number = SDF_MAX_POINTS) => /* wgsl */ `
${PRELUDE_3D}
struct SdfBody { position: vec4f, rotation: vec4f, velocity: vec4f, omega: vec4f, lo: vec4f, hi: vec4f, anchorRotation: vec4f }
struct Point { x: vec4f, normal: vec4f, anchorA: vec4f, anchorB: vec4f, flags: vec4u, ellipsoidA: vec4f, ellipsoidB: vec4f }
struct Result { identity: vec4u, stats: vec4u, points: array<Point,${maxPoints}> }
// counts: bodies, pair capacity, cell table size, cache table size.
// offsets: next links, large head, pair counter, cache table. The next two words
// after the pair counter hold tested pairs and large-body count.
struct BroadParams { counts: vec4u, offsets: vec4u, distances: vec4f }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> broad: BroadParams;
@group(0) @binding(2) var<storage,read> bodies: array<SdfBody>;
@group(0) @binding(3) var<storage,read_write> grid: array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> pairs: array<vec4u>;
@group(0) @binding(5) var<storage,read> previous: array<Result>;
@group(0) @binding(6) var<storage,read> statics: array<u32>;
@group(0) @binding(7) var<storage,read> joints: array<Joint>;
@group(0) @binding(8) var<storage,read> filters: array<vec2u>;
@group(0) @binding(9) var<storage,read_write> counters: array<atomic<u32>>;
const EMPTY=0xffffffffu;
${IGNORE_PAIRS_WGSL}
fn mixHash(value: u32) -> u32 {
  var x=value; x=(x^(x>>16u))*0x7feb352du; x=(x^(x>>15u))*0x846ca68bu; return x^(x>>16u);
}
fn pairHash(a:u32,b:u32) -> u32 { return mixHash(a^mixHash(b)); }
fn cell(i:u32) -> vec3i {
  let p=(bodies[i].lo.xyz*0.5+bodies[i].hi.xyz*0.5)/broad.distances.x;
  // Saturation preserves correctness (aliases are AABB-tested), avoids int overflow
  // in neighbour offsets, but remote saturated regions can lose spatial efficiency.
  return vec3i(clamp(floor(p),vec3f(-1e9),vec3f(1e9)));
}
fn bucket(c:vec3i) -> u32 { return mixHash((bitcast<u32>(c.x)*0x8da6b343u)^(bitcast<u32>(c.y)*0xd8163841u)^(bitcast<u32>(c.z)*0xcb1ab31fu))&(broad.counts.z-1u); }
fn large(i:u32) -> bool { return any(bodies[i].hi.xyz-bodies[i].lo.xyz+vec3f(broad.distances.y)>vec3f(broad.distances.x)); }
@compute @workgroup_size(64) fn clearGrid(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x;
  if(i<arrayLength(&grid)) { atomicStore(&grid[i],EMPTY); }
  if(i<broad.counts.y) { pairs[i]=vec4u(EMPTY); }
  // These words are assigned only by their own invocation, never by a racing clear.
  if(i>=broad.offsets.z && i<=broad.offsets.z+2u) { atomicStore(&grid[i],0u); }
}
@compute @workgroup_size(64) fn buildGrid(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x; if(i>=broad.counts.x) { return; }
  var head=bucket(cell(i));
  if(large(i)) { head=broad.offsets.y; atomicAdd(&grid[broad.offsets.z+2u],1u); }
  let old=atomicExchange(&grid[head],i); atomicStore(&grid[broad.offsets.x+i],old);
}
@compute @workgroup_size(64) fn cachePairs(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x; if(i>=broad.counts.y) { return; }
  let p=previous[i].identity;
  if(p.x>=broad.counts.x || p.x<=p.y) { return; }
  var h=pairHash(p.x,p.y)&(broad.counts.w-1u);
  for(var probe=0u;probe<2u*broad.counts.w;probe++) {
    let result=atomicCompareExchangeWeak(&grid[broad.offsets.w+h],EMPTY,i);
    if(result.exchanged) { return; }
    if(result.old_value!=EMPTY) { h=(h+1u)&(broad.counts.w-1u); }
  }
  atomicOr(&counters[C_OVERFLOW],1u);
}
fn previousSlot(a:u32,b:u32) -> u32 {
  var h=pairHash(a,b)&(broad.counts.w-1u);
  for(var probe=0u;probe<broad.counts.w;probe++) {
    let i=atomicLoad(&grid[broad.offsets.w+h]); if(i==EMPTY) { return EMPTY; }
    if(all(previous[i].identity.xy==vec2u(a,b))) { return i; }
    h=(h+1u)&(broad.counts.w-1u);
  }
  return EMPTY;
}
fn candidate(i:u32,j:u32) {
  let a=max(i,j); let b=min(i,j); atomicAdd(&grid[broad.offsets.z+1u],1u);
  let ba=bodies[a]; let bb=bodies[b]; let fa=filters[a]; let fb=filters[b];
  if((ba.velocity.w!=0.0&&bb.velocity.w!=0.0) || (fa.x&fb.y)==0u || (fb.x&fa.y)==0u ||
    any(ba.lo.xyz>bb.hi.xyz+vec3f(broad.distances.y)) || any(bb.lo.xyz>ba.hi.xyz+vec3f(broad.distances.y)) || ignored(a,b)) { return; }
  let slot=atomicAdd(&grid[broad.offsets.z],1u);
  if(slot>=broad.counts.y) {
    atomicOr(&counters[C_OVERFLOW],1u); atomicAdd(&counters[C_PAIRS],1u); return;
  }
  // The hash is a sampling key; exact (a,b) is the identity, even on hash collision.
  pairs[slot]=vec4u(a,b,pairHash(a,b),previousSlot(a,b));
}
@compute @workgroup_size(64) fn findPairs(@builtin(global_invocation_id) id:vec3u) {
  let i=id.x; if(i>=broad.counts.x) { return; }
  let isLarge=large(i);
  if(!isLarge) {
    let c=cell(i);
    for(var z=-1;z<=1;z++) { for(var y=-1;y<=1;y++) { for(var x=-1;x<=1;x++) {
      let neighbor=c+vec3i(x,y,z); var j=atomicLoad(&grid[bucket(neighbor)]);
      while(j!=EMPTY) {
        if(i>j && all(cell(j)==neighbor)) { candidate(i,j); }
        j=atomicLoad(&grid[broad.offsets.x+j]);
      }
    } } }
  }
  var j=atomicLoad(&grid[broad.offsets.y]);
  while(j!=EMPTY) {
    if(!isLarge || i>j) { candidate(i,j); }
    j=atomicLoad(&grid[broad.offsets.x+j]);
  }
}
`;
export const sdfBroadphaseWGSL = sdfBroadphaseSource();
