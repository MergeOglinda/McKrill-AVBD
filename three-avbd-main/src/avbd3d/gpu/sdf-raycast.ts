// Bounded GPU sphere tracing against the same immutable field atlas as collisions.
import { RAYCAST_DISPATCH_WGSL } from '../../lib/raycast.ts';
import { SDF_FIELDS_WGSL } from './wgsl-sdf-fields.ts';
import type { SdfAtlas } from './sdf-layout.ts';

/** Conservative global bound on |gradient(phi)|, including trilinear voxel interpolation.
 * Computed once from the packed snapshot, never from mutable user fields during queries. */
export function sdfRayLipschitz(atlas: SdfAtlas, header: number): number {
  const u=atlas.data, f=new Float32Array(u.buffer), h=header*4;
  let bound=1;
  if(u[h]===7) {
    const [offset,nx,ny,nz]=u.subarray(h+4,h+8), cell=f[h+11];
    const slopes=[0,0,0];
    for(let z=0;z<nz;z++) for(let y=0;y<ny;y++) for(let x=0;x<nx;x++) {
      const at=offset*4+4*(x+nx*(y+ny*z));
      if(x+1<nx)slopes[0]=Math.max(slopes[0],Math.abs(f[at+4]-f[at])/cell);
      if(y+1<ny)slopes[1]=Math.max(slopes[1],Math.abs(f[at+4*nx]-f[at])/cell);
      if(z+1<nz)slopes[2]=Math.max(slopes[2],Math.abs(f[at+4*nx*ny]-f[at])/cell);
    }
    // Outside the grid the field adds distance to its clamped sample.
    bound=Math.hypot(...slopes)+1;
  } else if(u[h]===6) {
    const k=Math.abs(f[h+23]);
    const radius=Math.hypot(...[0,1,2].map(i=>Math.max(Math.abs(f[h+8+i]),Math.abs(f[h+12+i]))));
    bound=1+k*(u[h+24]===2?7*Math.sqrt(3):radius);
  }
  const result=Math.fround(bound*1.001);
  if(!Number.isFinite(result))throw new Error('SDF ray slope bound exceeds float32 range');
  return result;
}

export const sdfRaycastWGSL = /* wgsl */ `
struct Body { position: vec4f, rotation: vec4f, velocity: vec4f, omega: vec4f, lo: vec4f, hi: vec4f, anchorRotation: vec4f }
struct Ray { o: vec4f, d: vec4f }
struct Params { bodies: u32, rays: u32, ignored: u32, mask: u32, ignore: array<vec4u,4> }
struct Hit { t: f32, n: vec3f }
@group(0) @binding(0) var<storage,read> bodies: array<Body>;
@group(0) @binding(1) var<storage,read> atlas: array<vec4u>;
@group(0) @binding(2) var<storage,read> rays: array<Ray>;
@group(0) @binding(3) var<storage,read_write> best: array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> hits: array<vec4f>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage,read> filters: array<vec2u>;
const NONE=0xffffffffu;
const MISS=Hit(-1.0,vec3f(0));
${SDF_FIELDS_WGSL}
fn ignored(b: u32) -> bool {
  if((filters[b].x&params.mask)==0u) { return true; }
  for(var i=0u;i<params.ignored;i++) { if(params.ignore[i/4u][i%4u]==b) { return true; } }
  return false;
}
fn worldNormal(b: u32, n: vec3f) -> vec3f { return n; }
fn trace(b: u32, r: Ray) -> Hit {
  let body=bodies[b]; var enter=0.0; var leave=r.o.w;
  if(leave<0.0 || dot(r.d.xyz,r.d.xyz)<0.5) { return MISS; }
  // Clip in field space: a rotated world AABB includes points beyond the domain
  // used for the warped-field slope bound. Parameter t remains in world units.
  let h=bodyHeader(body);
  let lo=bitcast<vec4f>(atlas[h+2u]).xyz; let hi=bitcast<vec4f>(atlas[h+3u]).xyz;
  let origin=localPoint(body,r.o.xyz); let direction=invrot(body.rotation,r.d.xyz)/body.position.w;
  for(var axis=0u;axis<3u;axis++) {
    let d=direction[axis]; let o=origin[axis];
    // Uniform scale can make a valid local direction arbitrarily small.
    if(d==0.0) {
      if(o<lo[axis] || o>hi[axis]) { return MISS; }
    } else {
      let a=(lo[axis]-o)/d; let c=(hi[axis]-o)/d;
      enter=max(enter,min(a,c)); leave=min(leave,max(a,c));
    }
  }
  if(enter>leave) { return MISS; }
  let extent=hi-lo;
  let tolerance=max(1e-6, min(extent.x,min(extent.y,extent.z))*1e-4)*body.position.w;
  var t=enter;
  for(var i=0u;i<512u;i++) {
    let s=worldField(body,r.o.xyz+r.d.xyz*t);
    if(s.phi<=tolerance) {
      var normal=unit(s.gradient);
      if(t==0.0 && s.phi<=0.0) { normal=-r.d.xyz; }
      return Hit(t,normal);
    }
    let next=t+s.phi/body.lo.w*0.9;
    if(next<=t || next>leave) { return MISS; }
    t=next;
  }
  // A bounded query that failed to converge is a miss, never a fabricated hit.
  return MISS;
}
${RAYCAST_DISPATCH_WGSL}
`;
