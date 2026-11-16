import { ExtrudeGeometry, Shape } from 'three';
import { bakeVoxelSdfAsync, curvatureSeeds, type VoxelField } from '../lib/sdf.ts';
import { World, type Body, type Vec3 } from '../lib/world.ts';

export const MODULE = .15;
export const DEPTH = .32;
export const TEETH = [20, 12, 28] as const;
export const CENTERS: Vec3[] = [[-2.4, 0, 0], [0, 0, 0], [3 * Math.cos(Math.PI / 6), 1.5, 0]];
export const COLORS = [0xeac46a, 0x74bed0, 0xd57968];
const pressure = 25 * Math.PI / 180;
const involute = (radius: number, base: number) => {
  const t = Math.sqrt(Math.max(0, (radius / base) ** 2 - 1));
  return t - Math.atan(t);
};

export interface Gear {
  teeth: number;
  geometry: ExtrudeGeometry;
  field: VoxelField;
  massProperties: { mass: number; moments: Vec3 };
}

/** Shared module and pressure angle; backlash leaves space for the voxel interpolation. */
export function gearProfile(teeth: number): [number, number][] {
  const pitch = MODULE * teeth / 2, base = pitch * Math.cos(pressure);
  const root = pitch - 1.25 * MODULE, tip = pitch + MODULE;
  const half = Math.PI / (2 * teeth) - .012 / pitch;
  const flank = (r: number) => half + involute(pitch, base) - involute(r, base);
  const points: [number, number][] = [];
  const point = (r: number, a: number) => points.push([r * Math.cos(a), r * Math.sin(a)]);
  for (let tooth = 0; tooth < teeth; tooth++) {
    const a = tooth * 2 * Math.PI / teeth;
    point(root, a - Math.PI / teeth);
    point(root, a - flank(Math.max(base, root)));
    for (let j = 0; j <= 8; j++) {
      const r = Math.max(base, root) + (tip - Math.max(base, root)) * j / 8;
      point(r, a - flank(r));
    }
    for (let j = 1; j <= 4; j++) point(tip, a - flank(tip) + 2 * flank(tip) * j / 4);
    for (let j = 7; j >= 0; j--) {
      const r = Math.max(base, root) + (tip - Math.max(base, root)) * j / 8;
      point(r, a + flank(r));
    }
    point(root, a + flank(Math.max(base, root)));
  }
  return points;
}

/** Integrals of the extruded polygon, density 1, about its symmetry/principal axes. */
export function gearMass(points: [number, number][]) {
  let area = 0, xx = 0, yy = 0;
  for (let i = 0; i < points.length; i++) {
    const [x, y] = points[i], [u, v] = points[(i + 1) % points.length];
    const cross = x * v - u * y;
    area += cross / 2;
    xx += (x * x + x * u + u * u) * cross / 12;
    yy += (y * y + y * v + v * v) * cross / 12;
  }
  const mass = area * DEPTH;
  return { mass, moments: [yy * DEPTH + mass * DEPTH ** 2 / 12, xx * DEPTH + mass * DEPTH ** 2 / 12, (xx + yy) * DEPTH] as Vec3 };
}

export async function prepareGears(progress: (text: string) => void = () => {}) {
  const gears: Gear[] = [];
  for (const teeth of TEETH) {
    progress(`Baking ${teeth}-tooth gear…`);
    const points = gearProfile(teeth), shape = new Shape();
    points.forEach(([x, y], i) => i ? shape.lineTo(x, y) : shape.moveTo(x, y)); shape.closePath();
    const geometry = new ExtrudeGeometry(shape, { depth: DEPTH, steps: 1, bevelEnabled: false });
    geometry.translate(0, 0, -DEPTH / 2);
    const baked = await bakeVoxelSdfAsync(geometry, { resolution: 80 });
    if (!baked.watertight) throw new Error(`Gear bake: ${baked.warnings.join(', ')}`);
    baked.field.seeds = curvatureSeeds(baked.field, { candidateCount: 1200 });
    gears.push({ teeth, geometry, field: baked.field, massProperties: gearMass(points) });
  }
  return gears;
}

export function initialAngles(): number[] {
  const angles = [0];
  for (let i = 1; i < TEETH.length; i++) {
    const a = Math.atan2(CENTERS[i][1] - CENTERS[i - 1][1], CENTERS[i][0] - CENTERS[i - 1][0]);
    angles.push(a + Math.PI + TEETH[i - 1] / TEETH[i] * (a - angles[i - 1]) - Math.PI / TEETH[i]);
  }
  return angles;
}
export const rotationZ = (a: number): [number, number, number, number] => [0, 0, Math.sin(a / 2), Math.cos(a / 2)];
export const angleZ = (body: Body) => 2 * Math.atan2(body.rotation[2], body.rotation[3]);
export const angleDelta = (a: number, b: number) => Math.atan2(Math.sin(a - b), Math.cos(a - b));

export async function createGearTrain(world: World, gears: Gear[], engaged = true) {
  const support = world.addBox({ size: [1, 1, 1], fixed: true, collidesWith: 0, group: 0 })!;
  const angles = initialAngles();
  const bodies = gears.map((gear, i) => world.addBox({ size: [1, 1, 1], position: CENTERS[i],
    rotation: rotationZ(angles[i]), fixed: i === 0, massProperties: gear.massProperties,
    friction: .05, collidesWith: engaged ? 1 : 0, group: 1 })!);
  // Two ball joints separated on z define a spindle, leaving rotation around z free.
  for (let i = 1; i < bodies.length; i++) for (const z of [-.65, .65]) {
    world.addJoint(support, bodies[i], { type: 'ball', anchorA: [CENTERS[i][0], CENTERS[i][1], z], anchorB: [0, 0, z] });
  }
  await world.enableSdfCollisions(bodies.map((body, i) => ({ body, field: gears[i].field })), {
    cacheSize: 4, randomSamples: 1, maxIterations: 60, tolerance: .001,
    epsilon: .0004, margin: .008, allowedPenetration: .001, alpha: .001,
    meshVertexContacts: true, refreshDeeperContacts: true,
  }, { maxPairs: 6 });
  return { bodies, support };
}

/** Drives only the input. moveTo keeps its contacts warm while changing speed/direction. */
export class GearDrive {
  angle = 0;
  velocity = 0;
  target = 0;
  automatic = true;
  speed = .5;
  readonly input: Body;
  constructor(input: Body) { this.input = input; }
  step(dt: number) {
    const desired = this.automatic ? this.speed : Math.max(-.85, Math.min(.85, (this.target - this.angle) * 8));
    this.velocity += Math.max(-1.8 * dt, Math.min(1.8 * dt, desired - this.velocity));
    this.angle += this.velocity * dt;
    this.input.moveTo(CENTERS[0], rotationZ(this.angle));
  }
}
