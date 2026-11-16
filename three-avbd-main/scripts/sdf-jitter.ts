// Diagnostic ensembles: deterministic per-body initial offsets (uniform ±10 µm) expose trajectory sensitivity.
import type { ReleasedScene } from '../src/sdf-released/scenes.ts';
export function jitterReleased(scene:ReleasedScene,seed:number):ReleasedScene{
  if(!seed)return scene;
  let state=seed*2654435761>>>0;const offset=()=>{state=Math.imul(state^state>>>15,0x2c1b3c6d)+0x6d2b79f5>>>0;return (state/2**32-.5)*2e-5;};
  return {...scene,bodies:scene.bodies.map(b=>b.original.static?b:{...b,original:{...b.original,position:b.original.position.map(v=>v+offset()) as typeof b.original.position}})};
}
