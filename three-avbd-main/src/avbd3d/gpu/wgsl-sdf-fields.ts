/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
// Field kernels ported from the pinned CPU reference. Native f32 WGSL evaluation
// can differ from the explicitly rounded oracle; hardware parity remains a gate.
export const SDF_FIELDS_WGSL = /* wgsl */ `
struct Sample { phi: f32, gradient: vec3f }
fn rot(q: vec4f, v: vec3f) -> vec3f { let t = 2.0 * cross(q.xyz, v); return v + q.w * t + cross(q.xyz, t); }
fn invrot(q: vec4f, v: vec3f) -> vec3f { return rot(vec4f(-q.xyz, q.w), v); }
fn unit(v: vec3f) -> vec3f { let l = length(v); if (l > 1e-12) { return v / l; } return vec3f(0,1,0); }
fn signnz(x: f32) -> f32 { return select(1.0, -1.0, x < 0.0); }
fn radial(p: vec3f, r: f32) -> vec3f { if (r > 1e-9) { return vec3f(p.x / r, 0, p.z / r); } return vec3f(1,0,0); }
fn boxField(p: vec3f, h: vec3f) -> Sample {
  let q = abs(p) - h; let outside = max(q, vec3f(0)); let d = length(outside); var g = vec3f(0);
  if (d > 0.0) { g = outside / d * vec3f(signnz(p.x), signnz(p.y), signnz(p.z)); }
  else { var axis = 2u; if (q.x > q.y) { if (q.x > q.z) { axis = 0u; } } else if (q.y > q.z) { axis = 1u; } g[axis] = signnz(p[axis]); }
  return Sample(d + min(max(q.x, max(q.y, q.z)), 0.0), g);
}
fn sphereField(p: vec3f, r: f32) -> Sample { let d = length(p); var g = vec3f(0,1,0); if (d > 1e-9) { g = p / d; } return Sample(d-r,g); }
fn cylinderField(p: vec3f, r: f32, h: f32) -> Sample {
  let radialLength = length(p.xz); let d = vec2f(radialLength-r, abs(p.y)-h); let outside = max(d,vec2f(0)); let l = length(outside); let e = radial(p,radialLength);
  var g = vec3f(0,signnz(p.y),0); if (d.x > d.y) { g = e; }
  if (l > 0.0) { g = e * (outside.x/l) + vec3f(0,signnz(p.y)*outside.y/l,0); }
  return Sample(l + min(max(d.x,d.y),0.0),g);
}
fn coneField(p: vec3f, radius: f32, h: f32) -> Sample {
  let qx = length(p.xz); let qy = p.y;
  let ca = vec2f(qx-min(qx,select(0.0,radius,qy<0.0)),abs(qy)-h); let k = vec2f(-radius,2.0*h);
  let t = clamp((qx*radius+(h-qy)*k.y)/dot(k,k),0.0,1.0); let cb = vec2f(qx,qy-h)+k*t;
  let s = select(1.0,-1.0,cb.x<0.0 && ca.y<0.0); let da=dot(ca,ca); let db=dot(cb,cb);
  var g=cb; if (da<db) { g=vec2f(ca.x,ca.y*signnz(qy)); }
  let l=length(g); if(l>1e-9) { g *= s/l; } else if(da<db) { g=vec2f(0,signnz(qy)); } else { g=normalize(vec2f(k.y,radius)); }
  return Sample(s*sqrt(min(da,db)),radial(p,qx)*g.x+vec3f(0,g.y,0));
}
fn torusField(p: vec3f, major: f32, minor: f32) -> Sample {
  let r=length(p.xz); let q=vec2f(r-major,p.y); let l=length(q); let e=radial(p,r); var g=e;
  if(l>1e-9) { g=e*(q.x/l)+vec3f(0,q.y/l,0); } return Sample(l-minor,g);
}
fn warpField(p: vec3f, h: vec3f, k: f32, kind: u32) -> Sample {
  if(kind==2u) { let s=sin(p*7.0); let c=cos(p*7.0); let b=boxField(p,h); return Sample(b.phi+k*s.x*s.y*s.z,b.gradient+7.0*k*vec3f(c.x*s.y*s.z,s.x*c.y*s.z,s.x*s.y*c.z)); }
  let angle=k*select(p.y,p.x,kind==0u); let c=cos(angle); let s=sin(angle); var q=vec3f(c*p.x-s*p.z,s*p.x+c*p.z,p.y);
  if(kind==0u) { q=vec3f(c*p.x-s*p.y,s*p.x+c*p.y,p.z); }
  let b=boxField(q,h); let g=b.gradient;
  if(kind==0u) { return Sample(b.phi,vec3f(dot(vec3f(c-k*q.y,s+k*q.x,0),g),dot(vec3f(-s,c,0),g),g.z)); }
  return Sample(b.phi,vec3f(dot(vec3f(c,s,0),g),dot(vec3f(-k*q.y,k*q.x,1),g),dot(vec3f(-s,c,0),g)));
}
fn shapeHash01(value: u32) -> f32 {
  var x=value; x=(x^(x>>16u))*0x7feb352du; x=(x^(x>>15u))*0x846ca68bu; x=x^(x>>16u); return f32(x&0xffffffu)/16777216.0;
}
fn spikePlateField(p: vec3f, primary: vec4f, secondary: vec4f) -> Sample {
  let h=primary.xyz; let count=i32(primary.w); let radius=secondary.x; let pitch=h*(2.0/primary.w);
  let ci=clamp(i32(floor((p.x+h.x)/pitch.x)),0,count-1); let cj=clamp(i32(floor((p.z+h.z)/pitch.z)),0,count-1);
  var result=boxField(p,h);
  for(var j=max(cj-1,0);j<=min(cj+1,count-1);j++) { for(var i=max(ci-1,0);i<=min(ci+1,count-1);i++) {
    let height=secondary.y+(secondary.z-secondary.y)*shapeHash01(17u*(u32(i)+u32(j)*u32(count))+11u);
    let center=vec3f(-h.x+(f32(i)+0.5)*pitch.x,h.y+0.5*height-0.5*radius,-h.z+(f32(j)+0.5)*pitch.z);
    let candidate=coneField(p-center,radius*(height+radius)/height,0.5*(height+radius));
    if(candidate.phi<result.phi) { result=candidate; }
  } }
  return result;
}
fn spikyWheelField(p: vec3f, primary: vec4f, spikes: f32) -> Sample {
  let radius=primary.x; let halfWidth=primary.y; let spikeRadius=primary.z; let spikeHeight=primary.w;
  var result=cylinderField(p,radius,halfWidth); let step=(2.0*3.1415927410125732)/spikes; let ring=halfWidth-spikeRadius;
  let nearest=i32(floor(atan2(p.z,p.x)/step+0.5));
  for(var k=nearest-1;k<=nearest+1;k++) {
    let a=f32(k)*step; let side=vec3f(-sin(a),0,cos(a)); let dir=vec3f(cos(a),0,sin(a));
    for(var j=0u;j<2u;j++) {
      let y=select(-ring,ring,j==1u); let r=p-dir*(radius+spikeHeight)-vec3f(0,y,0);
      let candidate=coneField(vec3f(dot(r,side),dot(r,dir),r.y),spikeRadius,spikeHeight);
      if(candidate.phi<result.phi) { result=Sample(candidate.phi,side*candidate.gradient.x+dir*candidate.gradient.y+vec3f(0,candidate.gradient.z,0)); }
    }
  }
  return result;
}
fn localField(h: u32, p: vec3f) -> Sample {
  let kind=atlas[h].x; let param=bitcast<vec4f>(atlas[h+5u]);
  switch(kind) {
    case 0u: { return boxField(p,param.xyz); }
    case 1u: { return sphereField(p,param.x); }
    case 2u: { return sphereField(vec3f(p.x,p.y-clamp(p.y,-param.y,param.y),p.z),param.x); }
    case 3u: { return cylinderField(p,param.x,param.y); }
    case 4u: { return coneField(p,param.x,param.y); }
    case 5u: { return torusField(p,param.x,param.y); }
    case 6u: { return warpField(p,param.xyz,param.w,atlas[h+6u].x); }
    case 8u: { return spikePlateField(p,param,bitcast<vec4f>(atlas[h+6u])); }
    case 9u: { return spikyWheelField(p,param,bitcast<vec4f>(atlas[h+6u]).x); }
    default: {
      let grid=atlas[h+1u]; let dims=grid.yzw; let origin=bitcast<vec4f>(atlas[h+7u]).xyz; let cell=bitcast<vec4f>(atlas[h+2u]).w;
      let hi=origin+vec3f(dims-vec3u(1u))*cell; let q=clamp(p,origin,hi); let coord=(q-origin)/cell;
      let base=min(vec3u(coord),dims-vec3u(2u)); let t=coord-vec3f(base); var value=vec4f(0);
      for(var i=0u;i<8u;i++) {
        let offset=vec3u(i&1u,(i>>1u)&1u,(i>>2u)&1u); let v=base+offset;
        let w=select(vec3f(1)-t,t,offset!=vec3u(0));
        value += bitcast<vec4f>(atlas[grid.x+v.x+dims.x*(v.y+dims.y*v.z)])*(w.x*w.y*w.z);
      }
      let outside=p-q; let d=length(outside); var g=value.yzw; if(d>0.0) { g=unit(g+outside/d); }
      return Sample(value.x+d,g);
    }
  }
}
fn bodyHeader(body: Body) -> u32 { return bitcast<u32>(body.omega.w); }
fn localPoint(body: Body, p: vec3f) -> vec3f { return invrot(body.rotation,p-body.position.xyz)/body.position.w+bitcast<vec4f>(atlas[bodyHeader(body)+4u]).xyz; }
fn worldPoint(body: Body, p: vec3f) -> vec3f { return body.position.xyz+rot(body.rotation,(p-bitcast<vec4f>(atlas[bodyHeader(body)+4u]).xyz)*body.position.w); }
fn worldField(body: Body, p: vec3f) -> Sample { let s=localField(bodyHeader(body),localPoint(body,p)); return Sample(s.phi*body.position.w,rot(body.rotation,s.gradient)); }
`;
