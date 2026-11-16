// URLs are emitted by Vite; assets stay lazy and are loaded only for the selected scene.
import type { ReadAsset } from './scenes.ts';
const urls=import.meta.glob('../../tests/fixtures/oracle-sdf{,-grids,-surfaces}/*',{query:'?url',import:'default',eager:true}) as Record<string,string>;
export const readBrowserAsset:ReadAsset=async path=>{
  const url=urls[`../../tests/fixtures/${path}`];if(!url)throw new Error(`Missing reference asset: ${path}`);
  const response=await fetch(url);if(!response.ok)throw new Error(`Reference download failed: ${path}`);
  return response.arrayBuffer();
};
