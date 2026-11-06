using System;
using NUnit.Framework;
using Unity.Mathematics;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Compares the Sequential mode of the C# reference solver with trajectories recorded from the
    /// upstream C++ solver (see UpstreamGolden.cs for how they were produced and how the tolerances
    /// were chosen). Passing here means the port reproduces upstream to within floating-point noise.
    /// </summary>
    public class UpstreamParityTests
    {
        public static UpstreamGoldenCase[] Cases => UpstreamGolden.Cases;

        internal static ReferenceSolver BuildScene(string scene)
        {
            var solver = new ReferenceSolver();
            switch (scene)
            {
                case "ground": AvbdScenes.Ground(solver); break;
                case "stack": AvbdScenes.Stack(solver); break;
                case "rope": AvbdScenes.Rope(solver); break;
                case "spring": AvbdScenes.Spring(solver); break;
                case "pyramid": AvbdScenes.Pyramid(solver); break;
                default: throw new ArgumentException("Unknown scene " + scene);
            }
            return solver;
        }

        [TestCaseSource(nameof(Cases))]
        public void SequentialMatchesUpstream(UpstreamGoldenCase golden)
        {
            ReferenceSolver solver = BuildScene(golden.Scene);
            Assert.AreEqual(golden.BodyCount, solver.Bodies.Count, "body count of scene " + golden.Scene);

            solver.Params.mode = GaussSeidelMode.Sequential;
            for (int step = 0; step < golden.Step; step++)
                solver.Step();

            Assert.AreEqual(0, solver.Stats.solveFailures, "solveFailures");

            for (int i = 0; i < golden.BodyCount; i++)
            {
                AvbdBody body = solver.Bodies[i];
                int o = i * 13;
                string what = $"{golden} body {i}";

                var pos = new float3(golden.State[o + 0], golden.State[o + 1], golden.State[o + 2]);
                var rot = new float4(golden.State[o + 3], golden.State[o + 4], golden.State[o + 5], golden.State[o + 6]);
                var linVel = new float3(golden.State[o + 7], golden.State[o + 8], golden.State[o + 9]);
                var angVel = new float3(golden.State[o + 10], golden.State[o + 11], golden.State[o + 12]);

                AssertNear(pos, body.PositionLin, golden.PositionTolerance, what + " position");

                // q and -q are the same rotation.
                float4 q = body.PositionAng.value;
                float4 qFlipped = -q;
                float4 nearest = math.lengthsq(q - rot) <= math.lengthsq(qFlipped - rot) ? q : qFlipped;
                for (int k = 0; k < 4; k++)
                    Assert.AreEqual(rot[k], nearest[k], golden.RotationTolerance, what + " rotation[" + k + "]");

                AssertNear(linVel, body.VelocityLin, golden.VelocityTolerance, what + " linear velocity");
                AssertNear(angVel, body.VelocityAng, golden.VelocityTolerance, what + " angular velocity");
            }
        }

        static void AssertNear(float3 expected, float3 actual, float tol, string what)
        {
            Assert.AreEqual(expected.x, actual.x, tol, what + " x");
            Assert.AreEqual(expected.y, actual.y, tol, what + " y");
            Assert.AreEqual(expected.z, actual.z, tol, what + " z");
        }
    }
}
