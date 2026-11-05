using NUnit.Framework;
using Unity.Mathematics;
using McKrill.Avbd.Reference;
using Random = Unity.Mathematics.Random;

namespace McKrill.Avbd.Tests
{
    public class ReferenceMathTests
    {
        const float Tight = 1e-5f;

        static void AssertNear(float3 expected, float3 actual, float tol, string what = "")
        {
            Assert.AreEqual(expected.x, actual.x, tol, what + " x");
            Assert.AreEqual(expected.y, actual.y, tol, what + " y");
            Assert.AreEqual(expected.z, actual.z, tol, what + " z");
        }

        static Vec6 RandomVec6(ref Random rng, float range)
        {
            return new Vec6(rng.NextFloat3(-range, range), rng.NextFloat3(-range, range));
        }

        // A = B B^T + shift * I is symmetric positive definite.
        static Mat6 RandomSpd(ref Random rng, float shift)
        {
            var b = new Mat6();
            for (int r = 0; r < 6; r++)
                for (int c = 0; c < 6; c++)
                    b.Set(r, c, rng.NextFloat(-1f, 1f));

            var a = new Mat6();
            for (int r = 0; r < 6; r++)
            {
                for (int c = 0; c < 6; c++)
                {
                    float s = (r == c) ? shift : 0f;
                    for (int k = 0; k < 6; k++) s += b.Get(r, k) * b.Get(c, k);
                    a.Set(r, c, s);
                }
            }
            return a;
        }

        // ---- Vec6 / Mat6 ------------------------------------------------------------------------

        [Test]
        public void Mat6_GetSet_RoundTripsAllEntries()
        {
            var m = new Mat6();
            for (int r = 0; r < 6; r++)
                for (int c = 0; c < 6; c++)
                    m.Set(r, c, r * 10 + c);

            for (int r = 0; r < 6; r++)
                for (int c = 0; c < 6; c++)
                    Assert.AreEqual(r * 10 + c, m.Get(r, c), 0f, $"entry ({r},{c})");
        }

        [Test]
        public void Mat6_Identity_MulIsNoOp()
        {
            var rng = new Random(1);
            Vec6 v = RandomVec6(ref rng, 5f);
            Vec6 r = Mat6.Identity.Mul(v);
            AssertNear(v.lin, r.lin, 0f);
            AssertNear(v.ang, r.ang, 0f);
        }

        [Test]
        public void Mat6_AddOuter_MatchesDefinition()
        {
            var rng = new Random(2);
            Vec6 j = RandomVec6(ref rng, 2f);
            var m = new Mat6();
            m.AddOuter(j, 3f);

            for (int r = 0; r < 6; r++)
                for (int c = 0; c < 6; c++)
                    Assert.AreEqual(3f * j[r] * j[c], m.Get(r, c), Tight, $"entry ({r},{c})");

            Assert.IsTrue(m.IsSymmetric(0f));
        }

        [Test]
        public void Mat6_AddCoupling_KeepsSymmetry()
        {
            var rng = new Random(3);
            var b = new float3x3(rng.NextFloat3(), rng.NextFloat3(), rng.NextFloat3());
            var m = Mat6.Identity;
            m.AddCoupling(b);
            Assert.IsTrue(m.IsSymmetric(0f));
        }

        [Test]
        public void LDLT_SolvesRandomSpdSystems()
        {
            var rng = new Random(12345);
            for (int trial = 0; trial < 200; trial++)
            {
                Mat6 a = RandomSpd(ref rng, 1f);
                Vec6 xTrue = RandomVec6(ref rng, 3f);
                Vec6 b = a.Mul(xTrue);

                Assert.IsTrue(a.TrySolveLDLT(b, out Vec6 x), $"trial {trial}");

                Vec6 residual = a.Mul(x) - b;
                Assert.Less(math.sqrt(Vec6.LengthSq(residual)), 1e-3f, $"residual, trial {trial}");
                Assert.Less(math.sqrt(Vec6.LengthSq(x - xTrue)), 1e-3f, $"error, trial {trial}");
            }
        }

        [Test]
        public void LDLT_DiagonalSystem_IsExact()
        {
            var a = new Mat6();
            for (int i = 0; i < 6; i++) a.Set(i, i, i + 1f);
            var b = new Vec6(new float3(1f, 2f, 3f), new float3(4f, 5f, 6f));

            Assert.IsTrue(a.TrySolveLDLT(b, out Vec6 x));
            for (int i = 0; i < 6; i++) Assert.AreEqual(1f, x[i], 1e-6f, $"component {i}");
        }

        [Test]
        public void LDLT_RejectsZeroNegativeAndNaN()
        {
            Assert.IsFalse(Mat6.Zero.TrySolveLDLT(Vec6.Zero, out _), "zero matrix");

            var neg = Mat6.Identity;
            neg.Set(3, 3, -1f);
            Assert.IsFalse(neg.TrySolveLDLT(Vec6.Zero, out _), "negative pivot");

            var nan = Mat6.Identity;
            nan.Set(2, 2, float.NaN);
            Assert.IsFalse(nan.TrySolveLDLT(Vec6.Zero, out _), "NaN pivot");
        }

        [Test]
        public void LDLT_ReadsOnlyLowerTriangle()
        {
            var rng = new Random(77);
            Mat6 a = RandomSpd(ref rng, 1f);
            Vec6 b = RandomVec6(ref rng, 2f);
            Assert.IsTrue(a.TrySolveLDLT(b, out Vec6 x1));

            Mat6 corrupted = a;
            for (int r = 0; r < 6; r++)
                for (int c = r + 1; c < 6; c++)
                    corrupted.Set(r, c, 999f); // upper triangle must not matter

            Assert.IsTrue(corrupted.TrySolveLDLT(b, out Vec6 x2));
            AssertNear(x1.lin, x2.lin, 0f);
            AssertNear(x1.ang, x2.ang, 0f);
        }

        [Test]
        public void LDLT_RigidBodyHessian_BlockStructureSolves()
        {
            // H = M/dt^2 on the linear block, rotated inertia/dt^2 on the angular block, plus one contact-like term.
            float dt = 1f / 60f;
            float mass = 2f;
            quaternion q = quaternion.AxisAngle(math.normalize(new float3(1f, 2f, 3f)), 0.7f);
            float3x3 inertia = AvbdQuat.RotatedDiagonal(q, new float3(0.5f, 1f, 2f));

            var h = new Mat6();
            h.AddScaledIdentity(mass / (dt * dt), 0f);
            h.AddAngular(inertia / (dt * dt));
            h.AddOuter(new Vec6(new float3(0f, 1f, 0f), new float3(0.3f, 0f, -0.2f)), 1000f);

            Vec6 f = new Vec6(new float3(1f, -2f, 0.5f), new float3(0.1f, 0.2f, -0.3f));
            Assert.IsTrue(h.TrySolveLDLT(f, out Vec6 dx));

            Vec6 back = h.Mul(dx);
            // Relative residual, since H entries are large (mass / dt^2 = 7200).
            float rel = math.sqrt(Vec6.LengthSq(back - f)) / math.sqrt(Vec6.LengthSq(f));
            Assert.Less(rel, 1e-4f);
        }

        // ---- Quaternion helpers -------------------------------------------------------------------

        [Test]
        public void Difference_OfEqualOrientations_IsZero()
        {
            quaternion q = quaternion.AxisAngle(math.normalize(new float3(0.3f, -0.5f, 0.8f)), 1.1f);
            AssertNear(float3.zero, AvbdQuat.Difference(q, q), Tight);
        }

        [Test]
        public void Difference_SmallRotation_ApproximatesRotationVector()
        {
            float3 axis = math.normalize(new float3(1f, 2f, -1f));
            float angle = 0.01f;
            quaternion a = quaternion.AxisAngle(axis, angle);
            AssertNear(axis * angle, AvbdQuat.Difference(a, quaternion.identity), 1e-6f);
        }

        [Test]
        public void Difference_IsInvariantToQuaternionSign()
        {
            quaternion a = quaternion.AxisAngle(math.normalize(new float3(0f, 1f, 1f)), 0.4f);
            quaternion b = quaternion.AxisAngle(math.normalize(new float3(1f, 0f, 1f)), -0.3f);
            quaternion negA = new quaternion(-a.value);

            AssertNear(AvbdQuat.Difference(a, b), AvbdQuat.Difference(negA, b), Tight);
        }

        [Test]
        public void Difference_IsAntisymmetricForSmallRotations()
        {
            quaternion a = quaternion.AxisAngle(math.normalize(new float3(0f, 1f, 1f)), 0.05f);
            quaternion b = quaternion.AxisAngle(math.normalize(new float3(1f, 0f, 1f)), -0.03f);
            AssertNear(-AvbdQuat.Difference(b, a), AvbdQuat.Difference(a, b), 1e-4f);
        }

        [Test]
        public void IntegrateAngular_KeepsUnitLength()
        {
            var rng = new Random(9);
            quaternion q = quaternion.identity;
            for (int i = 0; i < 1000; i++)
            {
                q = AvbdQuat.IntegrateAngular(q, rng.NextFloat3(-0.2f, 0.2f));
                Assert.AreEqual(1f, math.length(q.value), 1e-5f, $"step {i}");
            }
        }

        [Test]
        public void IntegrateAngular_SmallDelta_MatchesExactRotation()
        {
            quaternion q = quaternion.AxisAngle(math.normalize(new float3(1f, 1f, 0f)), 0.9f);
            float3 dw = new float3(0.004f, -0.006f, 0.003f);

            quaternion approx = AvbdQuat.IntegrateAngular(q, dw);
            quaternion exact = math.mul(quaternion.AxisAngle(math.normalize(dw), math.length(dw)), q);

            float3 err = AvbdQuat.Difference(approx, exact);
            Assert.Less(math.length(err), 1e-4f);
        }

        [Test]
        public void IntegrateAngular_ThenDifference_RecoversDelta()
        {
            quaternion q = quaternion.AxisAngle(math.normalize(new float3(-1f, 0.5f, 2f)), 2.0f);
            float3 dw = new float3(0.01f, 0.02f, -0.015f);
            quaternion q2 = AvbdQuat.IntegrateAngular(q, dw);
            AssertNear(dw, AvbdQuat.Difference(q2, q), 1e-4f);
        }

        [Test]
        public void RotatedDiagonal_IsSymmetricAndPreservesTrace()
        {
            quaternion q = quaternion.AxisAngle(math.normalize(new float3(1f, -1f, 2f)), 1.3f);
            float3 d = new float3(0.5f, 1.5f, 3f);
            float3x3 m = AvbdQuat.RotatedDiagonal(q, d);

            Assert.AreEqual(m[0][1], m[1][0], Tight);
            Assert.AreEqual(m[0][2], m[2][0], Tight);
            Assert.AreEqual(m[1][2], m[2][1], Tight);
            Assert.AreEqual(d.x + d.y + d.z, m[0][0] + m[1][1] + m[2][2], 1e-5f);
        }

        [Test]
        public void RotatedDiagonal_MatchesRotateScaleRotateBack()
        {
            quaternion q = quaternion.AxisAngle(math.normalize(new float3(2f, 0f, -1f)), 0.8f);
            float3 d = new float3(0.7f, 1.1f, 2.3f);
            float3 v = new float3(0.4f, -1.2f, 0.9f);

            float3 viaMatrix = math.mul(AvbdQuat.RotatedDiagonal(q, d), v);
            float3 viaRotations = math.rotate(q, d * math.rotate(math.conjugate(q), v));
            AssertNear(viaRotations, viaMatrix, 1e-5f);
        }

        [Test]
        public void Outer_And_Skew_MatchDefinitions()
        {
            float3 a = new float3(1f, 2f, 3f);
            float3 b = new float3(-2f, 0.5f, 4f);
            float3x3 o = AvbdQuat.Outer(a, b);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Assert.AreEqual(a[i] * b[j], o[j][i], 0f, $"outer ({i},{j})");

            AssertNear(math.cross(a, b), math.mul(AvbdQuat.Skew(a), b), Tight, "skew");
        }

        [Test]
        public void PoseDifference_CombinesPositionAndRotationError()
        {
            float3 pa = new float3(1f, 2f, 3f), pb = new float3(0.5f, 2f, 1f);
            quaternion qa = quaternion.AxisAngle(new float3(0f, 0f, 1f), 0.02f);
            Vec6 d = Vec6.PoseDifference(pa, qa, pb, quaternion.identity);
            AssertNear(pa - pb, d.lin, 0f);
            AssertNear(new float3(0f, 0f, 0.02f), d.ang, 1e-6f);
        }
    }
}
