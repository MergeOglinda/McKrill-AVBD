// Opt-in SDF contacts for the AVBD CPU reference. The default box oracle is unchanged.
import { Rigid, PENALTY_MIN, PENALTY_MAX } from '../body.ts';
import { Manifold } from '../manifold.ts';
import { vec3, mulv, transform, sub3, transpose, mat3 } from '../math.ts';
import type { Solver } from '../solver.ts';
import { AnalyticField, type Field } from './shapes.ts';
import { SdfBody } from './body.ts';
import { SdfManifold, collisionParams, type CollisionParams, type SdfContact } from './manifold.ts';
import { counters, type Counters } from './optimize.ts';
import { f, vec, minus, scale, dot, normalize, cross, inverse, rotate, Random, intersection, expand, valid, type Rotation } from './math.ts';

/** One AVBD force per SDF contact: nonconvex manifolds need independent normals. */
class PointForce extends Manifold {
  point: SdfContact;
  params: CollisionParams;
  constructor(solver: Solver, a: Rigid, b: Rigid, point: SdfContact, params: CollisionParams) {
    super(solver, a, b); this.point = point; this.params = params;
  }
  // Anchors are refreshed by the detector; initialize only warms their dual variables.
  override initialize(): boolean {
    const c = this.contacts[0];
    this.friction = Math.sqrt(this.bodyA!.friction * this.bodyB.friction);
    for (let i = 0; i < 3; i++) {
      c.lambda[i] *= this.solver.alpha * this.solver.gamma;
      c.penalty[i] = Math.max(PENALTY_MIN, Math.min(PENALTY_MAX, c.penalty[i] * this.solver.gamma));
    }
    return true;
  }
  refresh(a: SdfBody, b: SdfBody): void {
    const p = this.point, normal = scale(p.normal, -1);
    const tangent = normalize(Math.abs(normal[0]) > f(.57735) ? [normal[1], -normal[0], 0] : [0, normal[2], -normal[1]]);
    const second = cross(normal, tangent), old = this.contacts[0];
    const previousWorldForce = old ? mulv(vec3(), transpose(mat3(), this.basis), old.lambda) : vec3();
    this.basis.set([...normal, ...tangent, ...second]);
    const sa = a.sample(p.x), sb = b.sample(p.x);
    const xa = minus(p.x, scale(sa.gradient, f(sa.phi / Math.max(dot(sa.gradient, sa.gradient), 1e-8))));
    const xb = minus(p.x, scale(sb.gradient, f(sb.phi / Math.max(dot(sb.gradient, sb.gradient), 1e-8))));
    const ra = rotate(inverse(a.orientation), minus(xa, a.position)), rb = rotate(inverse(b.orientation), minus(xb, b.position));
    const c = old ?? { feature: p.id!, rA: vec3(), rB: vec3(), C0: vec3(), penalty: vec3(), lambda: vec3(), stick: false };
    if (old?.stick && this.params.refreshDeeperContacts) {
      const worldA = transform(vec3(), this.bodyA!.positionLin, this.bodyA!.positionAng, c.rA);
      const worldB = transform(vec3(), this.bodyB.positionLin, this.bodyB.positionAng, c.rB);
      const retained = dot(normal, vec(sub3(vec3(), worldA, worldB)));
      const moved = Math.max(Math.hypot(...minus(ra, vec(c.rA))), Math.hypot(...minus(rb, vec(c.rB))));
      if (moved > 3 * this.params.tolerance && p.separation < retained - this.params.tolerance) c.stick = false;
    }
    // Retain sticking anchors only for this cached contact identity, never by sorted rank.
    if (!old || !old.stick) { c.rA.set(ra); c.rB.set(rb); }
    mulv(c.lambda, this.basis, previousWorldForce);
    const worldA = transform(vec3(), this.bodyA!.positionLin, this.bodyA!.positionAng, c.rA);
    const worldB = transform(vec3(), this.bodyB.positionLin, this.bodyB.positionAng, c.rB);
    mulv(c.C0, this.basis, sub3(vec3(), worldA, worldB));
    // Constraint is negative for overlap. Match the SDF skin and allowed penetration.
    if (!old || !old.stick) { c.C0[1] = 0; c.C0[2] = 0; }
    c.C0[0] = (old?.stick ? c.C0[0] : p.separation) - this.params.margin + this.params.allowedPenetration;
    this.contacts = [c];
  }
}
interface Pair { a: Rigid; b: Rigid; manifold: SdfManifold; forces: Map<number, PointForce> }

/**
 * CPU reference collision bridge, not the public World API. Attach geometry to an existing
 * Rigid; its mass and diagonal inertia remain caller-owned AVBD properties. Imported fields
 * use their centerOfMass and supplied seeds. Solver gravity remains the existing z-up axis.
 */
export class SdfCollisionAdapter {
  readonly solver: Solver;
  readonly params: CollisionParams;
  readonly rng = new Random();
  stats: Counters = counters();
  private readonly fields = new Map<Rigid, SdfBody>();
  private readonly explicit = new Set<Rigid>();
  private pairs: Pair[] = [];
  private nextId = 1;
  private readonly detect = () => this.update();
  constructor(solver: Solver, params: Partial<CollisionParams> = {}) {
    if (solver.collisionDetection) throw new Error('Solver already has a collision backend');
    this.solver = solver; this.params = { ...collisionParams(), ...params };
    if (![0, 1, 2].includes(this.params.optimizer) || this.params.cacheSize < 0 || !Number.isInteger(this.params.cacheSize) || this.params.randomSamples < 0 || !Number.isInteger(this.params.randomSamples)) throw new Error('Invalid SDF collision parameters');
    solver.collisionDetection = this.detect;
  }
  attach(body: Rigid, field: Field): void {
    if (!this.solver.bodies.includes(body)) throw new Error('Body belongs to another solver');
    // Any old box manifold for this body must be replaced by SDF contacts.
    for (const force of [...body.forces]) if (force instanceof Manifold) force.destroy();
    this.dropPairs(body);
    this.fields.set(body, new SdfBody(field)); this.explicit.add(body);
  }
  dispose(): void {
    for (const p of this.pairs) for (const force of p.forces.values()) force.destroy();
    this.pairs = []; this.fields.clear(); this.explicit.clear();
    if (this.solver.collisionDetection === this.detect) this.solver.collisionDetection = null;
  }
  private dropPairs(body: Rigid): void {
    this.pairs = this.pairs.filter(p => {
      if (p.a !== body && p.b !== body) return true;
      for (const force of p.forces.values()) force.destroy(); return false;
    });
  }
  private sync(body: Rigid): SdfBody {
    let b = this.fields.get(body);
    if (!b) { b = new SdfBody(new AnalyticField({ kind: 'box', halfExtents: scale(vec(body.size), .5) })); this.fields.set(body, b); }
    b.position = vec(body.positionLin); b.orientation = Array.from(body.positionAng, f) as Rotation;
    b.velocity = vec(body.velocityLin); b.angularVelocity = vec(body.velocityAng); b.isStatic = body.mass <= 0;
    return b;
  }
  private update(): void {
    this.stats = counters(); const bodies = this.solver.bodies, kept: Pair[] = [];
    for (const body of this.fields.keys()) if (!bodies.includes(body)) { this.fields.delete(body); this.explicit.delete(body); }
    for (let i = bodies.length - 1; i >= 0; i--) for (let j = i - 1; j >= 0; j--) {
      const a = bodies[i], b = bodies[j];
      if (!this.explicit.has(a) && !this.explicit.has(b)) {
        const d = sub3(vec3(), a.positionLin, b.positionLin), r = a.radius + b.radius;
        if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r * r && !a.constrainedTo(b)) new Manifold(this.solver, a, b);
        continue;
      }
      if (a.mass <= 0 && b.mass <= 0) continue;
      if (a.forces.some(force => !(force instanceof Manifold) && (force.bodyA === b || force.bodyB === b))) continue;
      const sa = this.sync(a), sb = this.sync(b), margin = this.params.margin * .5;
      if (!valid(intersection(expand(sa.bounds, margin), expand(sb.bounds, margin)))) continue;
      let pair = this.pairs.find(p => p.a === a && p.b === b);
      if (!pair) pair = { a, b, manifold: new SdfManifold(sa, sb), forces: new Map() };
      pair.manifold.update(this.params, f(this.solver.dt), this.rng, this.stats);
      const alive = new Set<number>();
      for (const point of pair.manifold.points) {
        point.id ??= this.nextId++;
        if (!point.active) continue;
        alive.add(point.id);
        let force = pair.forces.get(point.id);
        if (!force) { force = new PointForce(this.solver, a, b, point, this.params); pair.forces.set(point.id, force); }
        force.point = point; force.refresh(sa, sb);
      }
      for (const [id, force] of pair.forces) if (!alive.has(id)) { force.destroy(); pair.forces.delete(id); }
      kept.push(pair);
    }
    for (const pair of this.pairs) if (!kept.includes(pair)) for (const force of pair.forces.values()) force.destroy();
    this.pairs = kept;
  }
}
