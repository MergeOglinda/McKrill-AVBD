using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Spheres: the narrow phase (sphere against sphere, sphere against box in both orders, a center inside a box), the contact handling of a rolling sphere,
    /// and a sphere on the ground. The reference solver has no spheres, so the expectations are written out by hand (the geometry is simple enough to
    /// do by hand, and Tools/HlslParity/sphere_check.cpp checks the HLSL against an independent double-precision computation on a million random pairs),
    /// and the GPU is checked against the CPU stage oracles (<see cref="CpuContacts"/>, <see cref="CpuStep"/>).
    /// Y-up, ground top at y = 0, like the Play Mode scenes.
    /// </summary>
    public class SphereTests
    {
        const int Capacity = 32;
        const int ContactSlots = 128;
        const float Tolerance = 1e-4f;

        // ---- the CPU oracle: geometry -----------------------------------------------------------------------------

        static BodyDescriptor Ground() => BodyDescriptor.Box(new float3(20f, 1f, 20f), 0f, new float3(0f, -0.5f, 0f));

        // Runs the oracle's narrow phase once (a predict first, as every step does: the narrow phase works on the poses of the start of the step).
        static CpuContacts Narrowphase(IList<BodyDescriptor> bodies, out SceneArrays arrays, out ConstraintRecord[] records)
        {
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            arrays = ConstraintTestScenes.FromDescriptors(bodies, Capacity);
            CpuContacts contacts = ContactTests.Oracle(arrays, out records);
            CpuPredict.Run(arrays.bodies, settings, arrays.bodyCount);
            contacts.Narrowphase(arrays.bodies, records, settings, arrays.bodyCount);
            return contacts;
        }

        static ConstraintRecord Contact(CpuContacts contacts, ConstraintRecord[] records, int a, int b, int k = 0)
        {
            return records[contacts.ContactBase + (int)ContactLayout.PairIndex(a, b) * ContactLayout.MaxContactsPerPair + k];
        }

        static void AssertContact(ConstraintRecord c, float3 normalBA, float constraintValue, float3 anchorA, float3 anchorB)
        {
            Assert.AreEqual(ContactLayout.SphereFeature, (int)c.ident.w, "feature key");
            ContactTests.AssertNear(normalBA, c.c0Ang.xyz, Tolerance, "normal (from B to A)");
            Assert.AreEqual(constraintValue, c.c0Lin.x, Tolerance, "normal row of C0 (minus the depth, plus the margin)");
            ContactTests.AssertNear(anchorA, c.anchorA.xyz, Tolerance, "anchor on A");
            ContactTests.AssertNear(anchorB, c.anchorB.xyz, Tolerance, "anchor on B");
        }

        [Test]
        public void Oracle_SphereOnTheGround_TouchesAtTheBottomOfTheSphere()
        {
            // A is the sphere (slot 1), 5 cm into the ground.
            CpuContacts contacts = Narrowphase(new[] { Ground(), BodyDescriptor.Sphere(0.5f, 1f, new float3(0f, 0.45f, 0f)) }, out _, out ConstraintRecord[] records);

            Assert.AreEqual(1u, contacts.PairCounts[(int)ContactLayout.PairIndex(1, 0)]);
            AssertContact(Contact(contacts, records, 1, 0), new float3(0f, 1f, 0f), -0.05f + ContactLayout.CollisionMargin,
                new float3(0f, -0.5f, 0f), new float3(0f, 0.5f, 0f));
        }

        [Test]
        public void Oracle_TwoSpheres_ContactAlongTheLineOfCenters()
        {
            CpuContacts contacts = Narrowphase(new[]
            {
                BodyDescriptor.Sphere(0.5f, 1f, new float3(0f, 0f, 0f)),
                BodyDescriptor.Sphere(0.5f, 1f, new float3(0.9f, 0f, 0f)),
            }, out _, out ConstraintRecord[] records);

            Assert.AreEqual(1u, contacts.PairCounts[(int)ContactLayout.PairIndex(1, 0)]);
            AssertContact(Contact(contacts, records, 1, 0), new float3(1f, 0f, 0f), -0.1f + ContactLayout.CollisionMargin,
                new float3(-0.5f, 0f, 0f), new float3(0.5f, 0f, 0f));
        }

        [Test]
        public void Oracle_ABoxOnASphere_ContactsWithTheBoxAsA()
        {
            // B is a static sphere (slot 0), A a box whose bottom face is 0.1 into it.
            CpuContacts contacts = Narrowphase(new[]
            {
                BodyDescriptor.Sphere(0.5f, 0f, float3.zero),
                BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0.9f, 0f)),
            }, out _, out ConstraintRecord[] records);

            Assert.AreEqual(1u, contacts.PairCounts[(int)ContactLayout.PairIndex(1, 0)]);
            AssertContact(Contact(contacts, records, 1, 0), new float3(0f, 1f, 0f), -0.1f + ContactLayout.CollisionMargin,
                new float3(0f, -0.5f, 0f), new float3(0f, 0.5f, 0f));
        }

        [Test]
        public void Oracle_ASphereWhoseCenterIsInsideABox_IsPushedOutThroughTheNearestFace()
        {
            // The box is 2 x 2 x 2 and static; the center of the sphere is 0.2 from the +x face.
            CpuContacts contacts = Narrowphase(new[]
            {
                BodyDescriptor.Box(new float3(2f), 0f, float3.zero),
                BodyDescriptor.Sphere(0.3f, 1f, new float3(0.8f, 0.1f, 0f)),
            }, out _, out ConstraintRecord[] records);

            Assert.AreEqual(1u, contacts.PairCounts[(int)ContactLayout.PairIndex(1, 0)]);
            // On the sphere: (0.5, 0.1, 0), on the box: (1, 0.1, 0); the sphere's surface is 0.3 + 0.2 behind the face.
            AssertContact(Contact(contacts, records, 1, 0), new float3(1f, 0f, 0f), -0.5f + ContactLayout.CollisionMargin,
                new float3(-0.3f, 0f, 0f), new float3(1f, 0.1f, 0f));
        }

        [Test]
        public void Oracle_SpheresThatAreApart_MakeNoContact()
        {
            CpuContacts contacts = Narrowphase(new[]
            {
                Ground(),
                BodyDescriptor.Sphere(0.5f, 1f, new float3(0f, 0.6f, 0f)),          // 10 cm above the ground
                BodyDescriptor.Sphere(0.5f, 1f, new float3(1.2f, 0.6f, 0f)),        // 0.2 from the other sphere
            }, out _, out _);

            Assert.AreEqual(0u, contacts.PairCounts[(int)ContactLayout.PairIndex(1, 0)], "sphere and ground");
            Assert.AreEqual(0u, contacts.PairCounts[(int)ContactLayout.PairIndex(2, 1)], "sphere and sphere");
            Assert.AreEqual(0u, contacts.PairCounts[(int)ContactLayout.PairIndex(2, 0)], "sphere and ground");
        }

        [Test]
        public void Oracle_ASphereContact_KeepsNoAnchorsWhenItSticks()
        {
            // Two narrow phases in a row; the second merges with the manifold of the first. A box contact that stuck would keep its old anchors; a sphere
            // contact takes the new ones, because the point it touches has moved over the surface.
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            SceneArrays arrays = ConstraintTestScenes.FromDescriptors(new[] { Ground(), BodyDescriptor.Sphere(0.5f, 1f, new float3(0f, 0.45f, 0f)) }, Capacity);
            CpuContacts contacts = ContactTests.Oracle(arrays, out ConstraintRecord[] records);
            CpuPredict.Run(arrays.bodies, settings, arrays.bodyCount);
            contacts.Narrowphase(arrays.bodies, records, settings, arrays.bodyCount);

            int index = contacts.ContactBase + (int)ContactLayout.PairIndex(1, 0) * ContactLayout.MaxContactsPerPair;
            records[index].ident.z |= (uint)ConstraintFlags.Stick;
            records[index].anchorA = new float4(9f, 9f, 9f, records[index].anchorA.w); // the anchor the stuck contact would keep

            CpuPredict.Run(arrays.bodies, settings, arrays.bodyCount);
            contacts.Narrowphase(arrays.bodies, records, settings, arrays.bodyCount);

            Assert.Less(math.length(records[index].anchorA.xyz), 1f, "the anchor of a sphere contact is taken from the new narrow phase");
        }

        // ---- the GPU against the oracle -----------------------------------------------------------------------------

        // A scene with a sphere on the ground, two overlapping spheres, a rotated box with a sphere on it, a static sphere with a box and a small sphere
        // whose center is inside that box, and a pile of two spheres: sphere-sphere, sphere-box in both orders, a center inside a box and static spheres.
        static List<BodyDescriptor> MixedScene()
        {
            BodyDescriptor tiltedBox = BodyDescriptor.Box(new float3(1f), 1f, new float3(3f, 0.45f, 0f));
            tiltedBox.rotation = quaternion.AxisAngle(new float3(0f, 1f, 0f), math.radians(30f));
            BodyDescriptor slantedBox = BodyDescriptor.Box(new float3(1.2f, 0.4f, 0.8f), 1f, new float3(-3.2f, 1.05f, 0.1f));
            slantedBox.rotation = quaternion.AxisAngle(new float3(0f, 0f, 1f), math.radians(20f));

            return new List<BodyDescriptor>
            {
                BodyDescriptor.Box(new float3(30f, 1f, 30f), 0f, new float3(0f, -0.5f, 0f)),           // 0 ground
                BodyDescriptor.Sphere(0.5f, 1f, new float3(0f, 0.45f, 0f)),                            // 1 on the ground
                BodyDescriptor.Sphere(0.6f, 1f, new float3(0.9f, 0.55f, 0.1f)),                        // 2 against 1 and the ground
                tiltedBox,                                                                              // 3 a rotated box on the ground
                BodyDescriptor.Sphere(0.4f, 1f, new float3(3.4f, 0.9f, 0.2f)),                         // 4 on the box
                BodyDescriptor.Sphere(0.7f, 0f, new float3(-3f, 0.5f, 0f)),                            // 5 a static sphere
                slantedBox,                                                                             // 6 a box on the static sphere
                BodyDescriptor.Sphere(0.3f, 1f, new float3(-3f, 1.0f, 0.1f)),                          // 7 a small sphere with its center inside the box
                BodyDescriptor.Sphere(0.5f, 1f, new float3(6f, 0.55f, 0f)),                            // 8 on the ground
                BodyDescriptor.Sphere(0.5f, 1f, new float3(6.3f, 1.4f, 0f)),                           // 9 on sphere 8
            };
        }

        [Test]
        public void Gpu_Narrowphase_MatchesTheOracle_OnSpheresAndBoxes()
        {
            ContactTests.RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            List<BodyDescriptor> bodies = MixedScene();
            SceneArrays arrays = ConstraintTestScenes.FromDescriptors(bodies, Capacity);
            CpuContacts contacts = ContactTests.Oracle(arrays, out ConstraintRecord[] oracleRecords);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                foreach (BodyDescriptor d in bodies)
                    buffers.Spawn(d);
                buffers.ApplyEdits();
                var gpu = new AvbdSolver(buffers, settings, null, constraints);
                constraints.Rebuild();

                int spherePairs = 0;
                for (int round = 0; round < 3; round++)
                {
                    // The broadphase of the GPU feeds the narrow phase; the oracle tests every pair, so a sphere pair the broadphase wrongly dropped shows up as a missing manifold.
                    gpu.DispatchBroadphase();
                    gpu.DispatchPredict();
                    constraints.PrepareContacts();
                    gpu.DispatchNarrowphase();

                    CpuPredict.Run(arrays.bodies, settings, arrays.bodyCount);
                    contacts.Narrowphase(arrays.bodies, oracleRecords, settings, arrays.bodyCount);

                    ContactSnapshot snapshot = constraints.ReadContacts();
                    ConstraintRecord[] records = constraints.ReadContactRecords();
                    Assert.AreEqual(0, snapshot.DuplicatePairs, "round " + round + ": a pair holds two slots");
                    Assert.AreEqual(0, snapshot.Overflow, "round " + round + ": the pool ran out of slots");

                    int oraclePairs = 0;
                    for (int a = 1; a < arrays.bodyCount; a++)
                        for (int b = 0; b < a; b++)
                        {
                            int pair = (int)ContactLayout.PairIndex(a, b);
                            string what = "round " + round + " pair (" + a + ", " + b + ")";
                            Assert.AreEqual(contacts.PairCounts[pair], snapshot.CountOf(a, b), what + " contact count");
                            if (contacts.PairCounts[pair] == 0u)
                                continue;

                            oraclePairs++;
                            int slot = snapshot.SlotOf(a, b);
                            for (int k = 0; k < (int)contacts.PairCounts[pair]; k++)
                            {
                                ConstraintRecord expected = oracleRecords[contacts.ContactBase + pair * ContactLayout.MaxContactsPerPair + k];
                                ContactTests.AssertRecordNear(expected, records[slot * ContactLayout.MaxContactsPerPair + k], Tolerance, what + " contact " + k);
                                if ((int)expected.ident.w == ContactLayout.SphereFeature)
                                    spherePairs++;
                            }
                        }
                    Assert.AreEqual(oraclePairs, snapshot.SlotsWithContacts, "round " + round + ": the GPU has manifolds the oracle does not");
                }
                Assert.GreaterOrEqual(spherePairs, 3 * 6, "the scene must make sphere contacts in every round, or nothing was tested");
            }
        }

        // A calmer scene for the steps: a rolling sphere, two spheres that touch, a sphere that lands on a static slab, and a sphere on a rotated box.
        static List<BodyDescriptor> StepScene()
        {
            BodyDescriptor rolling = BodyDescriptor.Sphere(0.5f, 1f, new float3(0f, 0.5f, 0f));
            rolling.linearVelocity = new float3(2f, 0f, 0.5f);
            BodyDescriptor falling = BodyDescriptor.Sphere(0.4f, 1f, new float3(-3.2f, 1.0f, 0f));
            falling.linearVelocity = new float3(0f, -6f, 0f);
            BodyDescriptor tiltedBox = BodyDescriptor.Box(new float3(1f), 1f, new float3(8f, 0.5f, 0f));
            tiltedBox.rotation = quaternion.AxisAngle(new float3(0f, 1f, 0f), math.radians(30f));

            return new List<BodyDescriptor>
            {
                BodyDescriptor.Box(new float3(30f, 1f, 30f), 0f, new float3(0f, -0.5f, 0f)),           // 0 ground
                rolling,                                                                                // 1
                BodyDescriptor.Sphere(0.5f, 1f, new float3(5f, 0.5f, 0f)),                             // 2 touches 3
                BodyDescriptor.Sphere(0.6f, 1f, new float3(5.95f, 0.6f, 0.1f)),                        // 3
                BodyDescriptor.Box(new float3(2f, 0.5f, 2f), 0f, new float3(-3f, 0.25f, 0f)),          // 4 static slab, top at y = 0.5
                falling,                                                                                // 5 lands on the slab
                tiltedBox,                                                                              // 6
                BodyDescriptor.Sphere(0.4f, 1f, new float3(8.2f, 1.38f, 0.1f)),                        // 7 on the box
            };
        }

        [Test]
        public void Gpu_Step_MatchesTheOracle_OnSpheresAndBoxes()
        {
            ContactTests.RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            List<BodyDescriptor> bodies = StepScene();
            SceneArrays arrays = ConstraintTestScenes.FromDescriptors(bodies, Capacity);
            CpuContacts contacts = ContactTests.Oracle(arrays, out ConstraintRecord[] oracleRecords);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                foreach (BodyDescriptor d in bodies)
                    buffers.Spawn(d);
                var solver = new AvbdSolver(buffers, settings, null, constraints);
                var state = new BodyStateRecord[Capacity];

                // A few steps only: rounding differences grow in contact dynamics.
                bool sawSphereContacts = false;
                for (int step = 1; step <= 8; step++)
                {
                    int failures = CpuStep.Run(arrays.bodies, oracleRecords, arrays.constraintCount, contacts, arrays.bodyCount, settings);
                    Assert.AreEqual(0, failures, "step " + step + ": the oracle must not fail a solve");
                    solver.Step();
                    Assert.AreEqual(0, constraints.ReadSolveFailures(), "step " + step + ": GPU solve failures");
                    sawSphereContacts |= constraints.ReadContacts().SlotsWithContacts > 0;

                    buffers.State.GetData(state);
                    for (int s = 0; s < arrays.bodyCount; s++)
                    {
                        string what = "step " + step + " slot " + s;
                        ContactTests.AssertNear(arrays.bodies.State[s].position.xyz, state[s].position.xyz, 5e-3f, what + " position");
                        ContactTests.AssertNear(arrays.bodies.State[s].rotation, state[s].rotation, 5e-3f, what + " rotation");
                    }
                }
                Assert.IsTrue(sawSphereContacts, "the scene never produced a contact, so nothing was tested");
            }
        }

        // ---- the GPU solver on spheres ------------------------------------------------------------------------------

        static void StepMany(AvbdSolver solver, int steps)
        {
            for (int i = 0; i < steps; i++)
                solver.Step();
        }

        [Test]
        public void Gpu_Step_ASphereComesToRestOnTheGround()
        {
            ContactTests.RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                buffers.Spawn(Ground());
                BodyHandle ball = buffers.Spawn(BodyDescriptor.Sphere(0.5f, 1f, new float3(0f, 2f, 0f)));
                var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default, null, constraints);

                StepMany(solver, 240);

                var state = new BodyStateRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];
                buffers.State.GetData(state);
                buffers.Motion.GetData(motion);
                Assert.AreEqual(0.5f, state[ball.Slot].position.y, 0.03f, "resting height: the radius above the ground");
                Assert.AreEqual(0f, state[ball.Slot].position.x, 0.01f);
                Assert.AreEqual(0f, state[ball.Slot].position.z, 0.01f);
                Assert.Less(math.length(motion[ball.Slot].linearVel.xyz), 0.05f, "linear velocity at rest");
                Assert.Less(math.length(motion[ball.Slot].angularVel.xyz), 0.1f, "angular velocity at rest");
                Assert.AreEqual(0, constraints.ReadSolveFailures());
            }
        }

        [Test]
        public void Gpu_Step_ASphereRestsOnABox()
        {
            ContactTests.RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                buffers.Spawn(Ground());
                buffers.Spawn(BodyDescriptor.Box(new float3(4f, 1f, 4f), 0f, new float3(0f, 0.5f, 0f))); // a static slab, top at y = 1
                BodyHandle ball = buffers.Spawn(BodyDescriptor.Sphere(0.5f, 1f, new float3(0.3f, 2.5f, 0.2f)));
                var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default, null, constraints);

                StepMany(solver, 240);

                var state = new BodyStateRecord[Capacity];
                buffers.State.GetData(state);
                Assert.AreEqual(1.5f, state[ball.Slot].position.y, 0.03f, "resting height: the slab's top plus the radius");
                Assert.AreEqual(0.3f, state[ball.Slot].position.x, 0.02f);
                Assert.AreEqual(0.2f, state[ball.Slot].position.z, 0.02f);
                Assert.AreEqual(0, constraints.ReadSolveFailures());
            }
        }

        [Test]
        public void Gpu_Step_ASphereRollsOnTheGround_WithoutSinkingOrHopping()
        {
            ContactTests.RequireCompute();
            const float radius = 0.5f;
            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                buffers.Spawn(Ground());
                BodyDescriptor d = BodyDescriptor.Sphere(radius, 1f, new float3(0f, radius, 0f));
                d.linearVelocity = new float3(4f, 0f, 0f);
                BodyHandle ball = buffers.Spawn(d);
                var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default, null, constraints);

                var state = new BodyStateRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];
                float lowest = float.MaxValue;
                float highest = float.MinValue;
                for (int step = 1; step <= 90; step++)
                {
                    solver.Step();
                    if (step < 15)
                        continue; // the first steps settle the contact
                    buffers.State.GetData(state);
                    lowest = math.min(lowest, state[ball.Slot].position.y);
                    highest = math.max(highest, state[ball.Slot].position.y);
                }

                buffers.State.GetData(state);
                buffers.Motion.GetData(motion);
                float3 v = motion[ball.Slot].linearVel.xyz;
                float3 w = motion[ball.Slot].angularVel.xyz;

                Assert.Greater(state[ball.Slot].position.x, 2.5f, "it keeps rolling: friction only slows it to 5/7 of the speed, and it does not stop");
                Assert.AreEqual(radius, lowest, 0.03f, "it does not sink into the ground");
                Assert.AreEqual(radius, highest, 0.03f, "it does not hop");
                // Rolling without slipping about the z axis: v = -w x r, so w.z = -v.x / r.
                Assert.Greater(v.x, 1.5f);
                Assert.AreEqual(-v.x / radius, w.z, 0.25f * math.abs(v.x / radius), "it rolls: the surface speed at the contact is about zero");
                Assert.Less(math.abs(w.x) + math.abs(w.y), 0.5f, "it spins about the z axis only");
                Assert.AreEqual(0, constraints.ReadSolveFailures());
            }
        }
    }
}
