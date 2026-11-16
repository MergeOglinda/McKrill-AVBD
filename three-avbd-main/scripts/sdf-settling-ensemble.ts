// Matched current-source settling acceptance, never a performance benchmark.
// Usage: nice -n 15 node --experimental-transform-types scripts/sdf-settling-ensemble.ts BASELINE_DIR OUT.json
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync,spawn} from 'node:child_process';
import {dirname,resolve} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {create,globals} from 'webgpu';
import {measuredFrame} from '../src/sdf-released/metrics.ts';
import {jitterReleased} from './sdf-jitter.ts';
import type {Sample,ReadAsset} from '../src/sdf-released/scenes.ts';
const script=fileURLToPath(import.meta.url),candidate=resolve(dirname(script),'..');
const frames=360,probeEvery=5,seeds=Array.from({length:11},(_,i)=>i);
const limits={6:{peak:.04,final:.005,energy:.01},9:{peak:.03,final:.01,energy:.01}} as const;
interface SceneRun {id:6|9;name:string;bodies:number;collisionParams:unknown;avbd:unknown;samples:Sample[]}
interface Run {variant:string;seed:number;adapter:unknown;scenes:SceneRun[]}
const digest=(b:Uint8Array|string)=>createHash('sha256').update(b).digest('hex');
async function manifest(root:string){
 const tracked=execFileSync('git',['ls-files','-z'],{cwd:root,encoding:'utf8'}).split('\0').filter(Boolean);
 const paths=tracked.filter(p=>/^(src\/|scripts\/|tests\/fixtures\/)/.test(p)||['package.json','pnpm-lock.yaml'].includes(p)).sort();
 const hashes:Record<string,string>={};for(const p of paths)hashes[p]=digest(await readFile(resolve(root,p)));
 return {head:execFileSync('git',['rev-parse','HEAD'],{cwd:root,encoding:'utf8'}).trim(),sourceSha256:digest(JSON.stringify(hashes)),files:hashes};
}
async function worker(root:string,out:string,seed:number){
 if(!seeds.includes(seed))throw new Error('Invalid jitter seed');
 Object.assign(globalThis,globals);const gpu=create([]),adapter=await gpu.requestAdapter();if(!adapter)throw new Error('No WebGPU adapter');
 const device=await adapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:adapter.limits.maxStorageBuffersPerShaderStage,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize,maxBufferSize:adapter.limits.maxBufferSize}});
 const errors:string[]=[];device.onuncapturederror=e=>errors.push(e.error.message);
 const {loadReleasedScene,instantiateReleased,releasedSettings}=await import(pathToFileURL(resolve(root,'src/sdf-released/scenes.ts')).href) as typeof import('../src/sdf-released/scenes.ts');
 const read:ReadAsset=async path=>{const b=await readFile(resolve(root,'tests/fixtures',path));return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength) as ArrayBuffer;};
 const report={seed,adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},scenes:[] as SceneRun[]};
 try{
  for(const id of [6,9] as const){
   const original=await loadReleasedScene(id,read),scene=jitterReleased(original,seed);
   try{
    const {world,bodies}=await instantiateReleased(scene,{device},'mesh-quality');
    try{
     const samples:Sample[]=[];
     for(let frame=1;frame<=frames;frame++){
      // Identical instrumentation for both sources: every-substep overflow, finite poses,
      // independent geometric probes, post-step energy, and a drained queue before yielding.
      samples.push(await measuredFrame(scene,world,bodies,frame,frame%probeEvery===0));
      if(errors.length)throw new Error(errors.join('\n'));
      await device.queue.onSubmittedWorkDone();await new Promise(r=>setTimeout(r,5));
     }
     report.scenes.push({id,name:id===6?'Pile of Chairs':'Bunny Pile',bodies:bodies.length,collisionParams:releasedSettings(scene,'mesh-quality').collision,
      avbd:{substeps:Math.round(scene.capture.solver.dt/world.dt),iterations:world.solver.params.iterations,dt:world.dt,alpha:world.solver.params.alpha,gamma:world.solver.params.gamma,massPenalty:world.solver.params.massPenalty},samples});
    }finally{world.destroy();}
   }finally{original.dispose();}
  }
  await writeFile(out,JSON.stringify(report)+'\n');
 }finally{device.destroy();}
}
function summarize(runs:Run[]){
 return ([6,9] as const).map(id=>{
  const byVariant=['baseline','candidate'].map(variant=>{
   const rows=runs.filter(r=>r.variant===variant).map(r=>{
    const scene=r.scenes.find(s=>s.id===id)!;if(!scene||scene.samples.length!==frames)throw new Error('Incomplete scene');
    const probes=scene.samples.filter(s=>s.maxVertexPenetration!==undefined),last=scene.samples.at(-1)!;
    if(probes.length!==frames/probeEvery||scene.samples.some((s,i)=>s.frame!==i+1||!Number.isFinite(s.kineticEnergy))||probes.some(s=>!Number.isFinite(s.maxVertexPenetration)))throw new Error('Invalid frame/probe record');
    const peak=Math.max(...probes.map(s=>s.maxVertexPenetration!)),final=last.maxVertexPenetration!,energy=last.kineticEnergy,l=limits[id];
    return {seed:r.seed,peak,final,energy,probesOver5mm:probes.filter(s=>s.maxVertexPenetration!>=.005).length,fails:peak>=l.peak||final>=l.final||energy>=l.energy};
   });
   const peaks=rows.map(r=>r.peak).sort((a,b)=>a-b);
   return {variant,rows,failures:rows.filter(r=>r.fails).length,probesOver5mm:rows.reduce((s,r)=>s+r.probesOver5mm,0),medianPeak:peaks.length?peaks[Math.floor(peaks.length/2)]:null};
  });
  const [a,b]=byVariant,complete=a.rows.length===seeds.length&&b.rows.length===seeds.length;
  return {id,byVariant,complete,accepted:complete&&b.failures<=a.failures&&b.probesOver5mm<=a.probesOver5mm&&b.medianPeak!<=a.medianPeak!};
 });
}
async function main(){
 const baseline=process.argv[2],output=process.argv[3];if(!baseline||!output)throw new Error('Usage: sdf-settling-ensemble.ts BASELINE_DIR OUT.json');
 const roots={baseline:resolve(baseline),candidate},out=resolve(output),parts=out+'.parts';
 await mkdir(parts,{recursive:true});
 const sources={baseline:await manifest(roots.baseline),candidate:await manifest(candidate)};
 const report={protocol:'Matched mesh-quality chair/bunny piles, jitter 0–10, 360 frames, probes every 5, 5ms idle per frame, 2s idle between processes; one GPU process at a time. Same current measuredFrame instrumentation for both source roots. Alternating variant order by seed. Not timing evidence.',
  frames,probeEvery,seeds,limits,decisionRule:'Per scene: failures, count of frame probes >=5mm and median peak must not increase. All seeds retained; no extended failure window.',
  node:process.version,sources,runs:[] as Run[],summary:[] as ReturnType<typeof summarize>,complete:false,accepted:false};
 await writeFile(out,JSON.stringify(report)+'\n');
 for(const seed of seeds)for(const variant of (seed%2?['candidate','baseline']:['baseline','candidate']) as ('baseline'|'candidate')[]){
  for(const v of ['baseline','candidate'] as const)if((await manifest(roots[v])).sourceSha256!==sources[v].sourceSha256)throw new Error(`${v} source changed during ensemble`);
  const path=resolve(parts,`${variant}-${seed}.json`);
  await new Promise<void>((ok,fail)=>{
   const child=spawn(process.execPath,['--experimental-transform-types','--no-warnings',script,'--worker',roots[variant],path,String(seed)],{stdio:'inherit'});
   const timer=setTimeout(()=>{child.kill('SIGKILL');fail(new Error(`${variant} seed ${seed} exceeded 180s`));},180_000);
   child.on('error',e=>{clearTimeout(timer);fail(e);});child.on('exit',(code,signal)=>{clearTimeout(timer);code===0?ok():fail(new Error(`${variant} seed ${seed}: ${code??signal}`));});
  });
  const run=JSON.parse(await readFile(path,'utf8')) as Omit<Run,'variant'>;
  if(run.seed!==seed)throw new Error('Seed mismatch');
  if(report.runs.length&&JSON.stringify(run.adapter)!==JSON.stringify(report.runs[0].adapter))throw new Error('GPU adapter changed during ensemble');
  const counterpart=report.runs.find(r=>r.seed===seed&&r.variant!==variant);
  if(counterpart)for(const scene of run.scenes){const other=counterpart.scenes.find(s=>s.id===scene.id)!;if(JSON.stringify([scene.collisionParams,scene.avbd])!==JSON.stringify([other.collisionParams,other.avbd]))throw new Error('Unmatched physical settings');}
  report.runs.push({variant,...run});report.summary=summarize(report.runs);
  await writeFile(out,JSON.stringify(report)+'\n');
  console.log(JSON.stringify({completed:report.runs.length,total:22,variant,seed,results:report.summary.map(s=>({id:s.id,...s.byVariant.find(v=>v.variant===variant)!.rows.find(r=>r.seed===seed)}))}));
  await new Promise(r=>setTimeout(r,2000));
 }
 for(const v of ['baseline','candidate'] as const)if((await manifest(roots[v])).sourceSha256!==sources[v].sourceSha256)throw new Error('Source changed before completion');
 report.complete=true;report.accepted=report.summary.every(s=>s.accepted);await writeFile(out,JSON.stringify(report)+'\n');
 console.log(JSON.stringify({complete:report.complete,accepted:report.accepted,summary:report.summary}));
}
if(process.argv[2]==='--worker')await worker(process.argv[3],process.argv[4],Number(process.argv[5]));else await main();
