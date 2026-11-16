import * as THREE from 'three/webgpu';
import { inspectQualificationWorld, qualificationOptions } from '../sdf-showcase/qualification.ts';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { BodyMesh, World, recommendedLimits, type Body, type Joint, type SdfBodyBinding, type Vec3 } from '../lib/index.ts';
import { showcaseShapes, type Kind } from './shapes.ts';
import { browserPerformance } from './performance.ts';
import './style.css';

const status = document.querySelector<HTMLOutputElement>('#status')!;
const preset = document.querySelector<HTMLSelectElement>('#preset')!;
const shapeSelect = document.querySelector<HTMLSelectElement>('#shape')!;
const contactToggle = document.querySelector<HTMLInputElement>('#contacts')!;
const pause = document.querySelector<HTMLButtonElement>('#pause')!;
const inputs = [...document.querySelectorAll<HTMLButtonElement | HTMLSelectElement | HTMLInputElement>('button,select,input')];
const fail = (error: unknown) => { status.textContent = String(error); console.error(error); };

async function main() {
  inputs.forEach(input => input.disabled = true);
  if (!navigator.gpu) throw new Error('WebGPU is unavailable. Open this demo in a WebGPU-enabled browser.');
  const qualification = qualificationOptions(new URLSearchParams(location.search));
  const initialCount = qualification?.count ?? 12, bodyLimit = Math.max(36, initialCount);
  const shapes = await showcaseShapes();
  const renderer = new THREE.WebGPURenderer({ antialias: true, requiredLimits: await recommendedLimits() });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  document.body.append(renderer.domElement);
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x101a22);
  scene.fog = new THREE.Fog(0x101a22, 20, 48);
  const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, .1, qualification ? 180 : 80);
  camera.position.set(10, 9, 13);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 3, 0); controls.enableDamping = true;
  const frameView = () => {
    const height = 1.4 + Math.floor((initialCount - 1) / (initialCount > 36 ? 64 : 3)) * 1.75;
    controls.target.set(0, qualification ? height / 2 : 3, 0);
    const distanceScale = qualification && initialCount > 12 ? Math.max(2, height / 9) : 1;
    camera.position.copy(new THREE.Vector3(10, 6, 13).multiplyScalar(distanceScale * Math.max(1, .85 / camera.aspect)).add(controls.target));
    camera.setViewOffset(innerWidth, innerHeight, innerWidth > 650 ? -150 : 0, innerWidth > 650 ? 0 : innerHeight * .16, innerWidth, innerHeight);
    camera.updateProjectionMatrix(); controls.update();
  };
  frameView();
  controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
  controls.minDistance = 4; controls.maxDistance = qualification ? 100 : 30; controls.maxPolarAngle = Math.PI * .48;
  scene.add(new THREE.HemisphereLight(0xc4e1ff, 0x34312b, 2));
  const sun = new THREE.DirectionalLight(0xffe3bd, 3); sun.position.set(4, 12, 6); sun.castShadow = true;
  Object.assign(sun.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12, far: 40 });
  if (qualification && initialCount > 12) {
    sun.position.set(12, 28, 18); Object.assign(sun.shadow.camera, { left: -24, right: 24, top: 24, bottom: -24, far: 100 });
  }
  sun.shadow.mapSize.set(1024, 1024); sun.shadow.normalBias = .03; scene.add(sun);
  const world = await World.create({ renderer, maxBodies: bodyLimit + 2, iterations: 12 });
  world.maxSubsteps = 2; world.readbackEvery = 0;
  const floor = world.addBox({ size: [24, .5, 24], position: [0, -.25, 0], fixed: true })!;
  // A collision-disabled kinematic handle lets the public joint API drag a picked point.
  const handle = world.addSphere({ radius: .05, fixed: true, collidesWith: 0, group: 0 })!;
  const floorMesh = new BodyMesh(world, { bodies: [floor], material: new THREE.MeshStandardNodeMaterial({ color: 0x243744, roughness: .85 }) });
  floorMesh.receiveShadow = true; scene.add(floorMesh);
  const grid = new THREE.GridHelper(24, 24, 0x42606c, 0x2c4653); grid.position.y = .006; scene.add(grid);
  const kinds = new Map<Body, Kind>();
  for (const kind of Object.keys(shapes) as Kind[]) {
    const shape = shapes[kind];
    const mesh = new BodyMesh(world, { bodies: body => kinds.get(body) === kind, geometry: shape.geometry,
      material: new THREE.MeshStandardNodeMaterial({ color: shape.color, roughness: .32, metalness: .22 }) });
    mesh.castShadow = mesh.receiveShadow = true; scene.add(mesh);
  }
  const markers = new THREE.Group(); scene.add(markers);
  const dotGeometry = new THREE.SphereGeometry(.035, 6, 4);
  const dotMaterial = new THREE.MeshBasicNodeMaterial({ color: 0xffdc77, depthTest: false });
  const lineMaterial = new THREE.LineBasicNodeMaterial({ color: 0xffdc77, depthTest: false });
  let busy = false, paused = true, inspecting = false, held = false, grab: Joint | null = null;
  let renderNeeded = true, debugNeeded = false, simulated = false, pickPending = false;
  controls.addEventListener('change', () => { renderNeeded = true; });
  let generation = 0, distance = 0, lastDebug = 0, lastFrame = 0, frame = 0, debugCount = 0;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const target = new THREE.Vector3();
  function clearMarkers() {
    for (const child of [...markers.children]) {
      markers.remove(child);
      if (child instanceof THREE.Line) child.geometry.dispose();
    }
  }
  function endDrag() { held = false; grab?.remove(); grab = null; controls.enabled = true; report(); }
  function report() {
    status.textContent = `${kinds.size} / ${bodyLimit} bodies · ${paused ? 'paused' : 'live'}${contactToggle.checked ? ` · ${debugCount} contact points` : ''}`;
    pause.textContent = paused ? 'Play' : 'Pause';
    document.querySelector<HTMLButtonElement>('#spawn')!.disabled = busy || kinds.size >= bodyLimit;
  }
  function setPaused(value: boolean) { paused = value; lastFrame = performance.now(); if (value) endDrag(); report(); }
  async function change(reset: boolean) {
    if (busy || (!reset && kinds.size >= bodyLimit)) return;
    perf.reset('Press Play or measure this scene.');
    busy = true; generation++; endDrag(); inputs.forEach(input => input.disabled = true);
    while (inspecting) await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
    clearMarkers(); renderNeeded = true; debugNeeded = simulated = false;
    status.textContent = 'Updating collision fields…';
    try {
      if (reset) { for (const body of kinds.keys()) body.remove(); kinds.clear(); }
      const bindings: SdfBodyBinding[] = [];
      const count = reset ? initialCount : Math.min(1, bodyLimit - kinds.size);
      const kind = (reset ? preset.value : shapeSelect.value) as Kind;
      const width = initialCount > 36 ? 8 : 3;
      for (let i = 0; i < count; i++) {
        const n = reset ? i : kinds.size;
        const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(.15 * (n % 3), n * .51, .17 * (n % 2))).toArray() as [number, number, number, number];
        const body = world.addBox({ size: [1, 1, 1], density: 2, friction: .5,
          position: reset ? initialCount > 36 ? [(i % width - 3.5) * 2.1, 1.4 + Math.floor(i / (width * width)) * 1.75, (Math.floor(i / width) % width - 3.5) * 2.1] : [(i % 3 - 1) * 2.1, 1.4 + Math.floor(i / 3) * 1.75, (i % 2 - .5) * .7] : [Math.sin(n) * 2, 8, Math.cos(n) * 2], rotation })!;
        kinds.set(body, kind); bindings.push({ body, field: shapes[kind].field });
      }
      if (world.sdfCollisionsEnabled) await world.updateSdfCollisions(bindings);
      else await world.enableSdfCollisions(bindings, {}, { maxPairs: qualification ? Math.max(703, bodyLimit * 32) : 703 });
      shapeSelect.value = kind; debugCount = 0; report();
    } catch (error) { setPaused(true); fail(error); }
    finally {
      busy = false; lastFrame = performance.now(); inputs.forEach(input => input.disabled = false);
      document.querySelector<HTMLButtonElement>('#spawn')!.disabled = kinds.size >= bodyLimit;
      document.querySelector<HTMLButtonElement>('#save-performance')!.disabled = !perf.report;
    }
  }
  const perf = browserPerformance({ canvas: renderer.domElement, device: world.device,
    context: () => ({ demo: 'interactive', preset: preset.value, dynamicBodies: kinds.size,
      bodyKinds: Object.fromEntries((['rings', 'warps', 'gears'] as const).map(kind => [kind, [...kinds.values()].filter(value => value === kind).length])),
      contacts: contactToggle.checked, iterations: 12, fixedDt: world.dt, maxSubsteps: world.maxSubsteps,
      maxPairs: world.sdfCollisionCapacity, pixelRatio: renderer.getPixelRatio(), shadows: renderer.shadowMap.enabled }),
    start: () => { if (busy || inspecting || pickPending || (qualification && (kinds.size !== initialCount || !world.sdfCollisionsEnabled || world.sdfCollisionsNeedUpdate))) return false; setPaused(false); return true; },
    inspect: () => inspectQualificationWorld(world),
    pause: () => setPaused(true) });
  if (qualification) {
    const kind = new URLSearchParams(location.search).get('preset');
    if (kind && ['rings', 'warps', 'gears'].includes(kind)) preset.value = kind;
    const note = document.querySelector('aside > details p');
    if (note) note.textContent = `${initialCount} initial bodies · ${bodyLimit}-body qualification limit. Same showcase fields and box inertia; larger cases use an 8 × 8 layered grid.`;
  }
  await change(true);
  document.querySelector('#reset')!.addEventListener('click', () => void change(true));
  preset.addEventListener('change', () => void change(true));
  document.querySelector('#spawn')!.addEventListener('click', () => void change(false));
  pause.addEventListener('click', () => { perf.cancel('Capture cancelled by playback change.'); setPaused(!paused); });
  document.querySelector('#view')!.addEventListener('click', () => { frameView(); renderNeeded = true; });
  contactToggle.addEventListener('change', () => { perf.reset('Contact overlay changed. Measure again when ready.'); generation++; clearMarkers(); renderNeeded = debugNeeded = true; report(); });
  function ray(event: PointerEvent) {
    const bounds = renderer.domElement.getBoundingClientRect();
    pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, 1 - (event.clientY - bounds.top) / bounds.height * 2);
    raycaster.setFromCamera(pointer, camera); return raycaster.ray;
  }
  // Reserve left gestures for asynchronous SDF picking; right-drag always orbits.
  renderer.domElement.addEventListener('pointerdown', event => {
    if (event.button !== 0 || busy || pickPending) return;
    const r = ray(event).clone(); held = true; controls.enabled = false; pickPending = true;
    renderer.domElement.setPointerCapture(event.pointerId);
    void (async () => {
      while (inspecting) await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      if (!held || busy) { pickPending = false; controls.enabled = true; return; }
      inspecting = true;
      try {
        const hit = await world.raycast(r.origin.toArray() as Vec3, r.direction.toArray() as Vec3, { ignore: [floor, handle] });
        if (!held || !hit?.body?.alive) { controls.enabled = true; return; }
        await world.read();
        if (!held) return;
        const body = hit.body;
        const local = new THREE.Vector3(...hit.point).sub(new THREE.Vector3(...body.position))
          .applyQuaternion(new THREE.Quaternion(...body.rotation).invert());
        distance = hit.distance; target.fromArray(hit.point); handle.set({ position: hit.point });
        grab = world.addJoint(handle, body, { type: 'ball', anchorB: local.toArray() as Vec3 });
        status.textContent = 'Grabbed · drag to move, release to drop';
      } catch (error) { endDrag(); fail(error); }
      finally { inspecting = false; pickPending = false; }
    })();
  }, true);
  renderer.domElement.addEventListener('pointermove', event => {
    if (grab) ray(event).at(distance, target);
  });
  addEventListener('pointerup', endDrag); addEventListener('pointercancel', endDrag);
  addEventListener('blur', endDrag);
  async function contacts() {
    inspecting = true; const revision = generation;
    try {
      // No simulation steps while these two snapshots are being copied/read.
      const points = await world.solver.readContactList(); await world.read();
      if (revision !== generation || !contactToggle.checked) return;
      clearMarkers(); debugCount = points.length;
      for (const point of points.slice(0, 256)) {
        const body = world.bodies.find(body => body.index === point.a); if (!body) continue;
        const p = new THREE.Vector3(...point.rA).applyQuaternion(new THREE.Quaternion(...body.rotation)).add(new THREE.Vector3(...body.position));
        const dot = new THREE.Mesh(dotGeometry, dotMaterial); dot.position.copy(p); markers.add(dot);
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([p, p.clone().addScaledVector(new THREE.Vector3(...point.normal), .28)]), lineMaterial);
        markers.add(line);
      }
      renderNeeded = true; report();
    } catch (error) { fail(error); }
    finally { inspecting = false; }
  }
  renderer.setAnimationLoop((time: number) => {
    if (document.hidden) return;
    perf.beginFrame(time);
    const dt = Math.max(0, (time - lastFrame) / 1000); lastFrame = time;
    let steps = 0;
    if (!busy && !inspecting && !paused) {
      if (grab) handle.moveTo(target.toArray() as Vec3);
      try { steps = world.update(dt); frame++; renderNeeded = debugNeeded = simulated = true; }
      catch (error) { perf.cancel('Capture stopped by a simulation error.'); setPaused(true); fail(error); }
    }
    if (simulated && debugNeeded && contactToggle.checked && !pickPending && !busy && !inspecting && time - lastDebug > 150) { lastDebug = time; debugNeeded = false; void contacts(); }
    perf.afterPhysics(steps);
    controls.update();
    if (renderNeeded && !busy) {
      renderer.render(scene, camera); perf.afterRender(); renderNeeded = false;
      if (!paused) perf.frame(time, steps * world.dt); else perf.idle();
    }
  });
  addEventListener('resize', () => { perf.reset('Viewport changed. Measure again when ready.'); camera.aspect = innerWidth / innerHeight; camera.setViewOffset(innerWidth, innerHeight, innerWidth > 650 ? -150 : 0, innerWidth > 650 ? 0 : innerHeight * .16, innerWidth, innerHeight); camera.updateProjectionMatrix(); renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5)); renderer.setSize(innerWidth, innerHeight); renderNeeded = true; });
  document.addEventListener('visibilitychange', () => { lastFrame = performance.now(); perf.idle(); if (document.hidden) { perf.cancel('Capture cancelled: tab was hidden.'); setPaused(true); } });
  // Small, explicit inspection hook for runtime smoke checks; no automatic benchmark.
  Object.assign(window, { sdfShowcase: { world, get frame() { return frame; }, get busy() { return busy || inspecting; }, get performance() { return perf.report; }, pause: () => { perf.cancel('Capture cancelled.'); setPaused(true); } } });
}
void main().catch(fail);
