/* Copyright (c) 2026 Chris Giles
 * Permission to use, copy, modify, distribute and sell this software and its
 * documentation for any purpose is hereby granted without fee, provided that
 * the above copyright notice appear in all copies. Chris Giles makes no
 * representations about the suitability of this software for any purpose.
 * It is provided "as is" without express or implied warranty.
 */
import { f, add, mul, sqrt, div, vec, plus, minus, scale, divide, rotate, inverse, cross, bounds, corners, type Vec, type Rotation, type Aabb } from './math.ts';
import type { Field, FieldSample } from './shapes.ts';
/** Position is center of mass; shape samples and seeds remain in the shape's local frame. */
export class SdfBody {
  field: Field;
  position: Vec;
  orientation: Rotation;
  scale: number;
  isStatic: boolean;
  velocity: Vec = [0, 0, 0];
  angularVelocity: Vec = [0, 0, 0];
  constructor(field: Field, position: Vec = [0, 0, 0], orientation: Rotation = [0, 0, 0, 1], scale = 1, isStatic = false) {
    if (!(f(scale) > 0 && Number.isFinite(f(scale)))) throw new Error('SDF scale must be finite and positive');
    if (!position.every(v => Number.isFinite(f(v))) || !orientation.every(v => Number.isFinite(f(v)))) throw new Error('SDF pose must be finite');
    const q = orientation.map(f) as Rotation, norm = sqrt(add(add(add(mul(q[0], q[0]), mul(q[1], q[1])), mul(q[2], q[2])), mul(q[3], q[3])));
    if (!(norm > 0) || !Number.isFinite(norm)) throw new Error('SDF orientation must be nonzero');
    this.field = field; this.position = vec(position); this.orientation = q.map(x => div(x, norm)) as Rotation;
    this.scale = f(scale); this.isStatic = isStatic;
  }
  toWorld(p: Vec): Vec { return plus(this.position, rotate(this.orientation, scale(minus(p, this.field.centerOfMass), this.scale))); }
  toLocal(p: Vec): Vec { return plus(divide(rotate(inverse(this.orientation), minus(p, this.position)), this.scale), this.field.centerOfMass); }
  sample(p: Vec): FieldSample {
    const s = this.field.sample(this.toLocal(p));
    return { phi: f(s.phi * this.scale), gradient: rotate(this.orientation, s.gradient) };
  }
  pointVelocity(p: Vec): Vec { return plus(this.velocity, cross(this.angularVelocity, minus(p, this.position))); }
  get bounds(): Aabb { return bounds(corners(this.field.bounds).map(p => this.toWorld(p))); }
  localBounds(b: Aabb): Aabb { return bounds(corners(b).map(p => this.toLocal(p))); }
}
