using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using Unity.Mathematics;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Behavior tests of the C# reference solver that do not need recorded data: resting contact,
    /// graph coloring, sequential-versus-colored agreement, collision basics, joints and removal.
    /// The sequential-versus-colored bounds were chosen from runs of a C++ prototype of the same
    /// colored ordering on the upstream scenes (worst observed difference 6e-4 on the stack and 1.5e-2 on
    /// the pyramid after 120 steps), with a few times of headroom.
    /// </summary>
    public class ReferenceSolverTests
    {
        static ReferenceSolver Run(string scene, GaussSeidelMode mode, int steps)
        {
            ReferenceSolver solver = UpstreamParityTests.BuildScene(scene);
            solver.Params.mode = mode;
            for (int i = 0; i < steps; i++)
                solver.Step();
            return solver;
        }

        static void AssertAllFinite(ReferenceSolver solver, string what)
        {
            foreach (AvbdBody b in solver.Bodies)
            {
                Assert.IsTrue(math.all(math.isfinite(b.PositionLin)), what + ": position of body " + b.Index);
                Assert.IsTrue(math.all(math.isfinite(b.PositionAng.value)), what + ": rotation of body " + b.Index);
                Assert.IsTrue(math.all(math.isfinite(b.VelocityLin)), what + ": linear velocity of body " + b.Index);
                Assert.IsTrue(math.all(math.isfinite(b.VelocityAng)), what + ": angular velocity of body " + b.Index);
            }
        }

        static float MaxPositionDifference(ReferenceSolver a, ReferenceSolver b)
        {
            float worst = 0f;
            for (int i = 0; i < a.Bodies.Count; i++)
                worst = math.max(worst, math.length(a.Bodies[i].PositionLin - b.Bodies[i].PositionLin));
            return worst;
        }

        static float MaxJointResidual(ReferenceSolver solver)
        {
            float worst = 0f;
            foreach (AvbdJoint j in solver.Forces.OfType<AvbdJoint>())
            {
                float3 a = j.BodyA != null ? AvbdQuat.Transform(j.BodyA.PositionLin, j.BodyA.PositionAng, j.rA) : j.rA;
                float3 b = AvbdQuat.Transform(j.BodyB.PositionLin, j.BodyB.PositionAng, j.rB);
                worst = math.max(worst, math.length(a - b));
            }
            return worst;
        }

        // ---- resting contact ---------------------------------------------------------------------------

        [TestCase(GaussSeidelMode.Sequential)]
        [TestCase(GaussSeidelMode.Colored)]
        public void BoxSettlesOnGround(GaussSeidelMode mode)
        {
            ReferenceSolver solver = Run("ground", mode, 240);

            AssertAllFinite(solver, mode.ToString());
            Assert.AreEqual(0, solver.Stats.solveFailures);

            // The slab top is at z = 0.5, the box is 1 tall, so its center rests near 1.0 (upstream: 0.991,
            // the collision margin and the soft contact let it sink about 9 mm).
            Assert.AreEqual(0.991f, solver.Bodies[1].PositionLin.z, 0.005f);
            Assert.Less(math.length(solver.Bodies[1].VelocityLin), 0.01f);
        }

        [Test]
        public void RestingContactKeepsWarmStartedManifold()
        {
            ReferenceSolver solver = Run("ground", GaussSeidelMode.Sequential, 120);

            AvbdManifold[] manifolds = solver.Forces.OfType<AvbdManifold>().ToArray();
            Assert.AreEqual(1, manifolds.Length, "one manifold between slab and box");

            AvbdManifold m = manifolds[0];
            Assert.AreEqual(4, m.NumContacts, "a flat box on a slab has four corner contacts");

            var keys = new HashSet<int>();
            for (int i = 0; i < m.NumContacts; i++)
            {
                Assert.IsTrue(keys.Add(m.Contacts[i].featureKey), "feature keys are unique");
                Assert.Less(m.Contacts[i].lambda.x, 0f, "a resting contact carries a (negative) normal force");
            }
        }

        // ---- sequential versus colored -----------------------------------------------------------------

        [Test]
        public void ColoredEqualsSequentialWhenThereIsNothingToColor()
        {
            // One dynamic body: both orderings do the same work.
            foreach (string scene in new[] { "ground", "spring" })
            {
                ReferenceSolver seq = Run(scene, GaussSeidelMode.Sequential, 120);
                ReferenceSolver col = Run(scene, GaussSeidelMode.Colored, 120);
                Assert.Less(MaxPositionDifference(seq, col), 1e-4f, scene);
            }
        }

        [Test]
        public void ColoredStackStaysCloseToSequential()
        {
            ReferenceSolver seq = Run("stack", GaussSeidelMode.Sequential, 120);
            ReferenceSolver col = Run("stack", GaussSeidelMode.Colored, 120);

            AssertAllFinite(col, "colored stack");
            Assert.Less(MaxPositionDifference(seq, col), 5e-3f);
            Assert.AreEqual(2, col.Stats.colors, "a column of boxes is 2-colorable");
        }

        [Test]
        public void ColoredPyramidStaysCloseToSequential()
        {
            ReferenceSolver seq = Run("pyramid", GaussSeidelMode.Sequential, 120);
            ReferenceSolver col = Run("pyramid", GaussSeidelMode.Colored, 120);

            AssertAllFinite(col, "colored pyramid");
            Assert.Less(MaxPositionDifference(seq, col), 0.05f);
            Assert.LessOrEqual(col.Stats.colors, 4);

            float maxSpeed = 0f;
            foreach (AvbdBody b in col.Bodies)
                maxSpeed = math.max(maxSpeed, math.length(b.VelocityLin));
            Assert.Less(maxSpeed, 1f, "the pyramid has settled");
        }

        [Test]
        public void ColoredRopeKeepsJointsAsTightAsSequential()
        {
            // The swinging rope is chaotic, so positions are not compared, only how well the joints hold.
            ReferenceSolver seq = Run("rope", GaussSeidelMode.Sequential, 120);
            ReferenceSolver col = Run("rope", GaussSeidelMode.Colored, 120);

            AssertAllFinite(col, "colored rope");
            Assert.Less(MaxJointResidual(seq), 0.05f);
            Assert.Less(MaxJointResidual(col), 0.05f);
            Assert.AreEqual(2, col.Stats.colors, "a chain is 2-colorable");
        }

        // ---- graph coloring ----------------------------------------------------------------------------

        [TestCase("ground")]
        [TestCase("stack")]
        [TestCase("rope")]
        [TestCase("spring")]
        [TestCase("pyramid")]
        public void ColoringIsValidAndComplete(string scene)
        {
            ReferenceSolver solver = Run(scene, GaussSeidelMode.Colored, 5);

            int dynamicBodies = solver.Bodies.Count(b => b.IsDynamic);
            Assert.AreEqual(dynamicBodies, solver.ColorGroups.Sum(g => g.Count), "every dynamic body is in exactly one group");
            Assert.AreEqual(solver.Stats.colors, solver.ColorGroups.Count);

            foreach (AvbdBody b in solver.Bodies)
            {
                if (!b.IsDynamic)
                {
                    Assert.AreEqual(-1, b.Color, "static bodies are not colored");
                    continue;
                }

                Assert.GreaterOrEqual(b.Color, 0);
                Assert.IsTrue(solver.ColorGroups[b.Color].Contains(b));

                foreach (AvbdForce f in b.Forces)
                {
                    if (!f.AffectsDynamics)
                        continue;
                    AvbdBody other = f.BodyA == b ? f.BodyB : f.BodyA;
                    if (other == null || !other.IsDynamic)
                        continue;
                    Assert.AreNotEqual(b.Color, other.Color, $"bodies {b.Index} and {other.Index} are joined by a force");
                }
            }
        }

        [Test]
        public void SequentialModeDoesNotColor()
        {
            ReferenceSolver solver = Run("stack", GaussSeidelMode.Sequential, 3);
            Assert.AreEqual(0, solver.Stats.colors);
            Assert.AreEqual(0, solver.ColorGroups.Count);
        }

        // ---- collision ---------------------------------------------------------------------------------

        static AvbdBody MakeBox(float3 size, float3 position, float density = 1f)
        {
            return new AvbdBody(size, density, 0.5f, position, float3.zero);
        }

        [Test]
        public void StackedBoxesProduceFourCornerContactsWithDownwardBasisNormal()
        {
            AvbdBody slab = MakeBox(new float3(100f, 100f, 1f), float3.zero, 0f);
            AvbdBody box = MakeBox(new float3(1f, 1f, 1f), new float3(0f, 0f, 0.99f));
            var contacts = new AvbdManifold.Contact[BoxCollision.MaxContacts];
            float3x3 basis = float3x3.identity;

            int n = BoxCollision.Collide(slab, box, contacts, ref basis);

            Assert.AreEqual(4, n);
            // Row 0 of the basis is the normal pointing from B (the box, above) to A (the slab).
            Assert.AreEqual(0f, AvbdMat3.Row(basis, 0).x, 1e-5f);
            Assert.AreEqual(0f, AvbdMat3.Row(basis, 0).y, 1e-5f);
            Assert.AreEqual(-1f, AvbdMat3.Row(basis, 0).z, 1e-5f);

            // The tangents are orthonormal and orthogonal to the normal.
            float3 n0 = AvbdMat3.Row(basis, 0);
            float3 t1 = AvbdMat3.Row(basis, 1);
            float3 t2 = AvbdMat3.Row(basis, 2);
            Assert.AreEqual(1f, math.length(t1), 1e-5f);
            Assert.AreEqual(1f, math.length(t2), 1e-5f);
            Assert.AreEqual(0f, math.dot(n0, t1), 1e-5f);
            Assert.AreEqual(0f, math.dot(n0, t2), 1e-5f);
            Assert.AreEqual(0f, math.dot(t1, t2), 1e-5f);

            // Contact points sit on the box's bottom face, in the box's local space at z = -0.5.
            for (int i = 0; i < n; i++)
                Assert.AreEqual(-0.5f, contacts[i].rB.z, 1e-4f);
        }

        [Test]
        public void SeparatedBoxesReturnNoContactsAndLeaveBasisUntouched()
        {
            AvbdBody slab = MakeBox(new float3(100f, 100f, 1f), float3.zero, 0f);
            AvbdBody box = MakeBox(new float3(1f, 1f, 1f), new float3(0f, 0f, 1.1f));
            var contacts = new AvbdManifold.Contact[BoxCollision.MaxContacts];
            float3x3 basis = new float3x3(7f);

            int n = BoxCollision.Collide(slab, box, contacts, ref basis);

            Assert.AreEqual(0, n);
            Assert.AreEqual(7f, basis.c0.x);
            Assert.AreEqual(7f, basis.c1.y);
            Assert.AreEqual(7f, basis.c2.z);
        }

        // ---- forces ------------------------------------------------------------------------------------

        [Test]
        public void IgnoreCollisionSuppressesTheManifold()
        {
            var solver = new ReferenceSolver();
            AvbdBody slab = solver.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, float3.zero);
            AvbdBody box = solver.AddBody(new float3(1f, 1f, 1f), 1f, 0.5f, new float3(0f, 0f, 0.9f));
            solver.AddIgnoreCollision(slab, box);

            solver.Step();

            Assert.AreEqual(0, solver.Stats.manifolds);
            Assert.AreEqual(0, solver.Forces.OfType<AvbdManifold>().Count());
            Assert.IsTrue(box.ConstrainedTo(slab));
        }

        [Test]
        public void ManifoldIsRemovedWhenBodiesSeparate()
        {
            var solver = new ReferenceSolver();
            AvbdBody slab = solver.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, float3.zero);
            AvbdBody box = solver.AddBody(new float3(1f, 1f, 1f), 1f, 0.5f, new float3(0f, 0f, 1.0f));
            solver.Step();
            Assert.AreEqual(1, solver.Forces.OfType<AvbdManifold>().Count(), "box touches the slab");

            // Teleport the box far away: the broadphase no longer pairs it, but the existing manifold
            // must be dropped by its own initialization (no contacts) the next step.
            box.PositionLin = new float3(0f, 0f, 20f);
            box.VelocityLin = float3.zero;
            solver.Step();

            Assert.AreEqual(0, solver.Forces.OfType<AvbdManifold>().Count());
            Assert.AreEqual(0, slab.Forces.Count);
            Assert.AreEqual(0, box.Forces.Count);
        }

        [Test]
        public void JointWithRigidAngleFracturesAndIsRemoved()
        {
            var solver = new ReferenceSolver();
            AvbdBody anchor = solver.AddBody(new float3(1f, 1f, 1f), 0f, 0.5f, new float3(0f, 0f, 10f));
            AvbdBody block = solver.AddBody(new float3(1f, 0.5f, 0.5f), 1f, 0.5f, new float3(1f, 0f, 10f));
            AvbdJoint joint = solver.AddJoint(
                anchor, block, new float3(0.5f, 0f, 0f), new float3(-0.5f, 0f, 0f),
                float.PositiveInfinity, float.PositiveInfinity, 0.01f);

            // Upstream breaks this joint on the second step.
            for (int i = 0; i < 10 && !joint.broken; i++)
                solver.Step();

            Assert.IsTrue(joint.broken);
            solver.Step(); // a broken joint is removed by the next initialization
            Assert.IsFalse(solver.Forces.Contains(joint));
            Assert.IsFalse(anchor.Forces.Contains(joint));
            Assert.IsFalse(block.Forces.Contains(joint));
        }

        [Test]
        public void JointWithoutAngularStiffnessNeverFractures()
        {
            var solver = new ReferenceSolver();
            AvbdBody anchor = solver.AddBody(new float3(1f, 1f, 1f), 0f, 0.5f, new float3(0f, 0f, 10f));
            AvbdBody block = solver.AddBody(new float3(1f, 0.5f, 0.5f), 1f, 0.5f, new float3(1f, 0f, 10f));
            AvbdJoint joint = solver.AddJoint(
                anchor, block, new float3(0.5f, 0f, 0f), new float3(-0.5f, 0f, 0f),
                float.PositiveInfinity, 0f, 0.01f);

            for (int i = 0; i < 60; i++)
                solver.Step();

            Assert.IsFalse(joint.broken, "fracture only looks at the angular multiplier");
            Assert.IsTrue(solver.Forces.Contains(joint));
        }

        [Test]
        public void WorldAnchoredJointHoldsABodyAtThePoint()
        {
            var solver = new ReferenceSolver();
            AvbdBody block = solver.AddBody(new float3(1f, 0.5f, 0.5f), 1f, 0.5f, new float3(1f, 0f, 10f));
            // bodyA == null pins the body-space point rB of the block to the world point rA.
            solver.AddJoint(null, block, new float3(0.5f, 0f, 10f), new float3(-0.5f, 0f, 0f));

            for (int i = 0; i < 120; i++)
                solver.Step();

            AssertAllFinite(solver, "pendulum");
            Assert.Less(MaxJointResidual(solver), 0.05f);
        }

        [Test]
        public void SpringSettlesAroundItsEquilibrium()
        {
            // Upstream spring scene: block m = 8 on a stiffness-100, rest-4 spring under g = 10 hangs
            // at 4 + m g / k = 4.8 below the anchor (z = 14 - 4.8 = 9.2). It is undamped, so it oscillates
            // around that; the time-averaged height must be near it.
            ReferenceSolver solver = UpstreamParityTests.BuildScene("spring");
            float sum = 0f;
            int samples = 0;
            for (int i = 0; i < 600; i++)
            {
                solver.Step();
                if (i >= 120) { sum += solver.Bodies[2].PositionLin.z; samples++; }
            }

            AssertAllFinite(solver, "spring");
            Assert.AreEqual(9.2f, sum / samples, 0.5f);
        }

        // ---- quaternion convention ---------------------------------------------------------------------

        [Test]
        public void DifferenceRawMatchesUpstreamAndOnlyDiffersBeyondHalfTurn()
        {
            quaternion b = quaternion.AxisAngle(math.normalize(new float3(1f, 2f, 3f)), 0.3f);
            quaternion a = math.mul(quaternion.AxisAngle(math.normalize(new float3(-1f, 0.5f, 2f)), 0.4f), b);

            float3 expected = math.mul(a, math.inverse(b)).value.xyz * 2f;
            float3 raw = AvbdQuat.DifferenceRaw(a, b);
            Assert.AreEqual(expected.x, raw.x, 1e-6f);
            Assert.AreEqual(expected.y, raw.y, 1e-6f);
            Assert.AreEqual(expected.z, raw.z, 1e-6f);

            // Small relative rotation: the canonicalized difference is identical.
            float3 canon = AvbdQuat.Difference(a, b);
            Assert.AreEqual(raw.x, canon.x, 1e-6f);
            Assert.AreEqual(raw.y, canon.y, 1e-6f);
            Assert.AreEqual(raw.z, canon.z, 1e-6f);

            // Relative rotation of 200 degrees: raw keeps the long way round, the canonical one flips.
            quaternion big = quaternion.AxisAngle(new float3(0f, 0f, 1f), math.radians(200f));
            float3 rawBig = AvbdQuat.DifferenceRaw(big, quaternion.identity);
            float3 canonBig = AvbdQuat.Difference(big, quaternion.identity);
            Assert.Greater(rawBig.z, 0f);
            Assert.Less(canonBig.z, 0f);
        }
    }
}
