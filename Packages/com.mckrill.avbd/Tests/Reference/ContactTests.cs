using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Contacts without a broadphase: stage N (box-box narrow phase over every pair of bodies, with the contact warm start), the contact
    /// terms of the primal stage and the contact dual update. The CPU oracle (<see cref="CpuContacts"/>) is checked against the reference
    /// solver's manifolds and trajectories, and the GPU solver against the oracle and the reference solver.
    ///
    /// The narrow phase is compared exactly: at the first step both start from the same bodies, so the manifolds must agree to rounding.
    /// Trajectories are compared with tolerances taken from how much upstream's own solver drifts when every position is nudged by 1e-6
    /// (C++ run on both scenes): the dropped box 7e-4 in position at step 60, the stack 7e-3 in position and 0.1 in velocity (at step 1,
    /// where the bottom box touches the ground exactly and the nudge decides whether it is in contact) and 8e-3 later. The bounds below
    /// are 5 to 15 times that. The pyramid is chaotic (1e-2 after 30 steps) and is not compared.
    /// </summary>
    public class ContactTests
    {
        const int Capacity = 64;
        const int ContactSlots = 256; // manifold slots of the GPU scenes: far more than their pairs need
        const float NarrowphaseTolerance = 1e-4f;

        // ---- the topology with contacts ----------------------------------------------------------------------------

        [Test]
        public void Topology_ListsTheContactsFirst_PairsAscending_ThenTheJointsNewestFirst()
        {
            // Bodies 0, 1, 2 are solved. Joint 0 joins 0 and 1. Pair (1, 0) has one contact, pair (2, 0) none, pair (2, 1) two.
            const int contactBase = 4;
            var records = new ConstraintRecord[contactBase];
            records[0] = ConstraintRecordFactory.Joint(0u, 1u, float3.zero, float3.zero, float.PositiveInfinity, 0f, float.PositiveInfinity, 1f);
            var pairCounts = new uint[3];
            pairCounts[ContactLayout.PairIndex(1, 0)] = 1u;
            pairCounts[ContactLayout.PairIndex(2, 1)] = 2u;

            var t = new ConstraintTopology();
            t.Build(records, 1, new[] { true, true, true }, 3, true, pairCounts, 3, contactBase);

            uint c10 = (uint)(contactBase + ContactLayout.PairIndex(1, 0) * ContactLayout.MaxContactsPerPair);
            uint c21 = (uint)(contactBase + ContactLayout.PairIndex(2, 1) * ContactLayout.MaxContactsPerPair);

            AssertList(t, 0, new[] { c10, 0u }, new[] { 1u, 1u });
            AssertList(t, 1, new[] { c10, c21, c21 + 1, 0u }, new[] { 0u, 2u, 2u, 0u });
            AssertList(t, 2, new[] { c21, c21 + 1 }, new[] { 1u, 1u });

            // The edges are 0-1 (the joint and a contact) and 1-2 (a contact); pair (2, 0) has none. A path needs two colors.
            Assert.AreEqual(2, t.ColorCount);
        }

        [Test]
        public void Topology_ListsAContactOnlyForItsSolvedBodies()
        {
            var pairCounts = new uint[3];
            pairCounts[ContactLayout.PairIndex(1, 0)] = 1u; // body 0 is static: only body 1 lists this contact
            pairCounts[ContactLayout.PairIndex(2, 0)] = 1u; // only body 2 lists this one
            pairCounts[ContactLayout.PairIndex(2, 1)] = 1u;

            var t = new ConstraintTopology();
            t.Build(new ConstraintRecord[0], 0, new[] { false, true, true }, 3, true, pairCounts, 3, 0);

            Assert.AreEqual(0, (int)t.Ranges[0].range.y, "a static body has no list");
            Assert.AreEqual(2, (int)t.Ranges[1].range.y, "the contact with the static body and the one with body 2");
            Assert.AreEqual(2, (int)t.Ranges[2].range.y, "the contact with the static body and the one with body 1");
            Assert.AreEqual(2, t.ColorCount, "the two solved bodies are neighbors; the static one is not colored");
        }

        [Test]
        public void Topology_LeavesAFracturedJointOut()
        {
            ConstraintRecord broken = ConstraintRecordFactory.Joint(0u, 1u, float3.zero, float3.zero, float.PositiveInfinity, float.PositiveInfinity, 1f, 1f);
            broken.ident.z |= (uint)ConstraintFlags.Broken;
            ConstraintRecord intact = ConstraintRecordFactory.Joint(1u, 2u, float3.zero, float3.zero, float.PositiveInfinity, float.PositiveInfinity, 1f, 1f);

            var t = new ConstraintTopology();
            t.Build(new[] { broken, intact }, 2, new[] { true, true, true }, 3);

            Assert.AreEqual(0, (int)t.Ranges[0].range.y, "the fractured joint is not listed");
            AssertList(t, 1, new[] { 1u }, new[] { 2u });
            AssertList(t, 2, new[] { 1u }, new[] { 1u });
            Assert.AreEqual(2, t.ColorCount, "only the intact joint is an edge");
        }

        static void AssertList(ConstraintTopology t, int body, uint[] ids, uint[] others)
        {
            int offset = (int)t.Ranges[body].range.x;
            Assert.AreEqual(ids.Length, (int)t.Ranges[body].range.y, "list length of body " + body);
            for (int i = 0; i < ids.Length; i++)
            {
                Assert.AreEqual(ids[i], t.List[offset + i], "constraint " + i + " of body " + body);
                Assert.AreEqual(others[i], t.ListOther[offset + i], "other body of constraint " + i + " of body " + body);
            }
        }

        // ---- the oracle against the reference solver -------------------------------------------------------------

        [TestCase("overlap")]
        [TestCase("stack")]
        public void Oracle_Narrowphase_MatchesTheReferenceManifolds_AtTheFirstStep(string scene)
        {
            ReferenceSolver solver = Colored(scene, out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(solver, Capacity);
            ConstraintRecord[] records;
            CpuContacts contacts = Oracle(arrays, out records);

            solver.Step();
            CpuPredict.Run(arrays.bodies, settings, arrays.bodyCount); // the narrow phase reads the start poses, which predict sets
            contacts.Narrowphase(arrays.bodies, records, settings, arrays.bodyCount);

            AssertManifoldsMatch(solver, contacts, records, scene, NarrowphaseTolerance);
        }

        [TestCase("ground", 90)]
        [TestCase("stack", 60)]
        public void Oracle_Step_MatchesTheReferenceSolver(string scene, int steps)
        {
            ReferenceSolver solver = Colored(scene, out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(solver, Capacity);
            ConstraintRecord[] records;
            CpuContacts contacts = Oracle(arrays, out records);

            bool sawContacts = false;
            for (int step = 1; step <= steps; step++)
            {
                solver.Step();
                int failures = CpuStep.Run(arrays.bodies, records, arrays.constraintCount, contacts, arrays.bodyCount, settings);
                Assert.AreEqual(0, failures, scene + " step " + step + ": the oracle must not fail a solve");
                sawContacts |= HasContacts(contacts);

                if (Array.IndexOf(Checkpoints(scene), step) < 0)
                    continue;
                for (int s = 0; s < arrays.bodyCount; s++)
                    AssertBodyNear(solver.Bodies[s], arrays.bodies, s, scene, step);
            }
            Assert.IsTrue(sawContacts, scene + ": the scene never produced a contact, so nothing was tested");
        }

        [Test]
        public void Oracle_BoxComesToRestOnTheGround()
        {
            ReferenceSolver solver = Colored("ground", out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(solver, Capacity);
            CpuContacts contacts = Oracle(arrays, out ConstraintRecord[] records);

            for (int step = 0; step < 200; step++)
                CpuStep.Run(arrays.bodies, records, arrays.constraintCount, contacts, arrays.bodyCount, settings);

            // The ground's top face is at z = 0.5 and the box is 1 high.
            float3 p = arrays.bodies.State[1].position.xyz;
            Assert.AreEqual(1f, p.z, 0.05f, "resting height");
            Assert.AreEqual(0f, p.x, 0.01f);
            Assert.AreEqual(0f, p.y, 0.01f);
            Assert.Less(math.length(arrays.bodies.Motion[1].linearVel.xyz), 0.05f, "velocity at rest");
        }

        [Test]
        public void Oracle_StackStaysUpright()
        {
            ReferenceSolver solver = Colored("stack", out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(solver, Capacity);
            CpuContacts contacts = Oracle(arrays, out ConstraintRecord[] records);

            for (int step = 0; step < 240; step++)
                CpuStep.Run(arrays.bodies, records, arrays.constraintCount, contacts, arrays.bodyCount, settings);

            for (int i = 1; i <= 10; i++)
            {
                float3 p = arrays.bodies.State[i].position.xyz;
                Assert.IsTrue(math.all(math.isfinite(p)), "box " + i + " position");
                Assert.AreEqual(0f, p.x, 0.1f, "box " + i + " x");
                Assert.AreEqual(0f, p.y, 0.1f, "box " + i + " y");
                Assert.Greater(p.z, i - 0.25f, "box " + i + " must not sink into the one below");
                Assert.Less(p.z, i + 0.25f, "box " + i + " must rest on the one below");
            }
        }

        [Test]
        public void Oracle_ForgetSlot_DropsTheManifoldsOfThatSlot()
        {
            ReferenceSolver solver = Colored("stack", out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(solver, Capacity);
            CpuContacts contacts = Oracle(arrays, out ConstraintRecord[] records);
            for (int step = 0; step < 5; step++)
                CpuStep.Run(arrays.bodies, records, arrays.constraintCount, contacts, arrays.bodyCount, settings);
            Assert.Greater(contacts.PairCounts[ContactLayout.PairIndex(1, 0)], 0u, "the first box touches the ground");

            contacts.ForgetSlot(1);

            Assert.AreEqual(0u, contacts.PairCounts[ContactLayout.PairIndex(1, 0)]);
            Assert.AreEqual(0u, contacts.PairCounts[ContactLayout.PairIndex(2, 1)]);
        }

        // ---- the GPU ---------------------------------------------------------------------------------------------

        [TestCase("overlap")]
        [TestCase("stack")]
        public void Gpu_Narrowphase_MatchesTheOracle(string scene)
        {
            RequireCompute();
            ReferenceSolver solver = Colored(scene, out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(solver, Capacity);
            CpuContacts contacts = Oracle(arrays, out ConstraintRecord[] oracleRecords);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                foreach (BodyDescriptor d in arrays.descriptors)
                    buffers.Spawn(d);
                buffers.ApplyEdits();
                var gpu = new AvbdSolver(buffers, settings, null, constraints);
                constraints.Rebuild();

                // Three narrow phases in a row, with a predict between: the later ones merge with the manifolds of the earlier ones.
                // Nothing solves in between, so the bodies only move along their velocities.
                for (int round = 0; round < 3; round++)
                {
                    gpu.DispatchBroadphase();
                    gpu.DispatchPredict();
                    constraints.PrepareContacts();
                    gpu.DispatchNarrowphase();

                    CpuPredict.Run(arrays.bodies, settings, arrays.bodyCount);
                    contacts.Narrowphase(arrays.bodies, oracleRecords, settings, arrays.bodyCount);

                    ContactSnapshot snapshot = constraints.ReadContacts();
                    ConstraintRecord[] records = constraints.ReadContactRecords();

                    // Same manifolds, in whatever slots the pool gave them.
                    Assert.AreEqual(0, snapshot.DuplicatePairs, scene + " round " + round + ": a pair holds two slots");
                    Assert.AreEqual(0, snapshot.Overflow, scene + " round " + round + ": the pool ran out of slots");
                    Assert.AreEqual(snapshot.SlotsWithContacts, snapshot.UsedSlots, scene + " round " + round + ": a slot without contacts must be free");
                    Assert.AreEqual(ContactSlots, snapshot.FreeSlots + snapshot.UsedSlots, scene + " round " + round + ": free + used slots");

                    int oraclePairs = 0;
                    for (int a = 1; a < arrays.bodyCount; a++)
                        for (int b = 0; b < a; b++)
                        {
                            int pair = (int)ContactLayout.PairIndex(a, b);
                            string what = scene + " round " + round + " pair (" + a + ", " + b + ")";
                            Assert.AreEqual(contacts.PairCounts[pair], snapshot.CountOf(a, b), what + " contact count");
                            if (contacts.PairCounts[pair] == 0u)
                                continue;

                            oraclePairs++;
                            int slot = snapshot.SlotOf(a, b);
                            for (int k = 0; k < (int)contacts.PairCounts[pair]; k++)
                                AssertRecordNear(
                                    oracleRecords[contacts.ContactBase + pair * ContactLayout.MaxContactsPerPair + k],
                                    records[slot * ContactLayout.MaxContactsPerPair + k], NarrowphaseTolerance, what + " contact " + k);
                        }
                    Assert.AreEqual(oraclePairs, snapshot.SlotsWithContacts, scene + " round " + round + ": the GPU has manifolds the oracle does not");
                }
            }
        }

        [TestCase("ground", 90)]
        [TestCase("stack", 60)]
        public void Gpu_Step_MatchesTheReferenceSolver(string scene, int steps)
        {
            RequireCompute();
            ReferenceSolver reference = Colored(scene, out AvbdSolverSettings settings);
            SceneArrays arrays = ConstraintTestScenes.FromReference(reference, Capacity);

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                foreach (BodyDescriptor d in arrays.descriptors)
                    buffers.Spawn(d);
                var solver = new AvbdSolver(buffers, settings, null, constraints);
                var state = new BodyStateRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];

                bool sawContacts = false;
                for (int step = 1; step <= steps; step++)
                {
                    reference.Step();
                    solver.Step();
                    sawContacts |= constraints.ReadContacts().SlotsWithContacts > 0;

                    if (Array.IndexOf(Checkpoints(scene), step) < 0)
                        continue;

                    buffers.State.GetData(state);
                    buffers.Motion.GetData(motion);
                    for (int s = 0; s < arrays.bodyCount; s++)
                    {
                        AvbdBody b = reference.Bodies[s];
                        string what = scene + " step " + step + " slot " + s;
                        AssertNear(b.PositionLin, state[s].position.xyz, PositionTolerance(scene), what + " position");
                        AssertNear(b.PositionAng.value, state[s].rotation, RotationTolerance(scene), what + " rotation");
                        AssertNear(b.VelocityLin, motion[s].linearVel.xyz, VelocityTolerance(scene), what + " linear velocity");
                        AssertNear(b.VelocityAng, motion[s].angularVel.xyz, VelocityTolerance(scene), what + " angular velocity");
                    }
                    Assert.AreEqual(0, constraints.ReadSolveFailures(), scene + " step " + step + ": GPU solve failures");
                }
                Assert.IsTrue(sawContacts, scene + ": the scene never produced a contact, so nothing was tested");
            }
        }

        [Test]
        public void Gpu_Step_BoxesJoinedByAJointDoNotCollide()
        {
            RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = float3.zero;

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                // Two overlapping boxes, and a third one overlapping the second.
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));
                BodyHandle b = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0.8f, 0f, 0f)));
                BodyHandle c = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(1.6f, 0f, 0f)));
                constraints.AddJoint(a, b, new float3(0.4f, 0f, 0f), new float3(-0.4f, 0f, 0f));
                var solver = new AvbdSolver(buffers, settings, null, constraints);

                solver.Step();

                ContactSnapshot snapshot = constraints.ReadContacts();
                Assert.AreEqual(0u, snapshot.CountOf(a.Slot, b.Slot), "joined bodies make no manifold");
                Assert.Greater(snapshot.CountOf(b.Slot, c.Slot), 0u, "bodies that are not joined do");
                Assert.AreEqual(0, constraints.ReadSolveFailures());
            }
        }

        [Test]
        public void Gpu_Narrowphase_ARecycledSlotDoesNotInheritContacts()
        {
            RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = float3.zero;

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                BodyHandle ground = buffers.Spawn(BodyDescriptor.Box(new float3(10f, 10f, 1f), 0f, new float3(0f, 0f, 0f)));
                BodyHandle box = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0.9f)));
                buffers.ApplyEdits();
                var solver = new AvbdSolver(buffers, settings, null, constraints);
                constraints.Rebuild();

                // Round 1: a new contact starts at the minimum penalty.
                NarrowphaseRound(buffers, constraints, solver);
                int first = ManifoldSlot(constraints, box, ground);
                Assert.AreEqual(settings.penaltyMin, constraints.ReadContactRecord(first, 0).penaltyLin.x, 1e-6f, "a new contact starts at the minimum penalty");

                // Pretend the solver grew the penalty. Round 2 keeps the contact and decays the penalty by gamma.
                ConstraintRecord grown = constraints.ReadContactRecord(first, 0);
                grown.penaltyLin = new float4(500f, 500f, 500f, 0f);
                constraints.WriteContactRecord(first, 0, grown);
                NarrowphaseRound(buffers, constraints, solver);
                Assert.AreEqual(first, ManifoldSlot(constraints, box, ground), "a pair that keeps touching keeps its slot");
                float kept = constraints.ReadContactRecord(first, 0).penaltyLin.x;
                Assert.Greater(kept, 100f, "the contact of a live pair keeps its penalty");

                // The box goes and a new one takes its slot, at the same place: it starts without history.
                buffers.Remove(box);
                BodyHandle replacement = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0.9f)));
                Assert.AreEqual(box.Slot, replacement.Slot, "the slot is reused");
                buffers.ApplyEdits();
                if (constraints.NeedsRebuild)
                    constraints.Rebuild();
                NarrowphaseRound(buffers, constraints, solver);
                first = ManifoldSlot(constraints, replacement, ground);
                Assert.AreEqual(settings.penaltyMin, constraints.ReadContactRecord(first, 0).penaltyLin.x, 1e-6f, "the new body does not inherit the penalty of the old one");
            }
        }

        // What the step does up to the end of the narrow phase.
        static void NarrowphaseRound(AvbdBodyBuffers buffers, AvbdConstraintBuffers constraints, AvbdSolver solver)
        {
            solver.DispatchBroadphase();
            solver.DispatchPredict();
            constraints.PrepareContacts();
            solver.DispatchNarrowphase();
        }

        // The manifold slot of the pair of two bodies: where the pool put its manifold.
        static int ManifoldSlot(AvbdConstraintBuffers constraints, BodyHandle a, BodyHandle b)
        {
            int slot = constraints.ReadContacts().SlotOf(a.Slot, b.Slot);
            Assert.GreaterOrEqual(slot, 0, "the pair has no manifold");
            return slot;
        }

        [Test]
        public void Gpu_Step_AFracturedJointStopsExcludingItsPairFromCollisions()
        {
            RequireCompute();
            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            settings.gravity = float3.zero;

            using (var buffers = new AvbdBodyBuffers(Capacity))
            using (var constraints = new AvbdConstraintBuffers(buffers, 4, null, ContactSlots))
            {
                // Two overlapping boxes joined at the middle of the overlap, with a rigid angle lock that breaks at once; b spins.
                BodyHandle a = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 0f, 0f)));
                BodyDescriptor spinning = BodyDescriptor.Box(new float3(1f), 1f, new float3(0.9f, 0f, 0f));
                spinning.angularVelocity = new float3(0f, 0f, 5f);
                BodyHandle b = buffers.Spawn(spinning);
                constraints.AddJoint(a, b, new float3(0.45f, 0f, 0f), new float3(-0.45f, 0f, 0f), float.PositiveInfinity, float.PositiveInfinity, 1e-3f);
                var solver = new AvbdSolver(buffers, settings, null, constraints);

                solver.Step();
                Assert.AreEqual(0u, constraints.ReadContacts().CountOf(a.Slot, b.Slot), "the joint keeps the pair from colliding while it holds");

                // The joint breaks on the GPU in some step k. In step k + 1 the broadphase still finds the pair joined (the reference solver
                // removes a broken joint at the initialize of that step), and the list build (stage G) after the narrow phase leaves the
                // joint out. From step k + 2 the pair is a candidate.
                for (int i = 0; i < 10 && constraints.BrokenCount == 0; i++)
                    solver.Step();
                Assert.AreEqual(1, constraints.BrokenCount, "the joint fractured");
                Assert.AreEqual(0u, constraints.ReadContacts().CountOf(a.Slot, b.Slot), "the step that sees the fracture still has the pair joined");

                solver.Step();
                Assert.Greater(constraints.ReadContacts().CountOf(a.Slot, b.Slot), 0u, "the boxes still overlap and now collide");
                solver.Step();
                Assert.Greater(constraints.ReadContacts().CountOf(a.Slot, b.Slot), 0u, "and keep colliding");
            }
        }

        [Test]
        public void Gpu_Constraints_RejectTooManyContactBodies()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(Capacity))
            {
                Assert.Throws<ArgumentOutOfRangeException>(() => new AvbdConstraintBuffers(buffers, 4, null, -1));
                Assert.Throws<ArgumentOutOfRangeException>(() => new AvbdConstraintBuffers(buffers, 4, null, AvbdConstraintBuffers.MaxContactManifolds + 1));
            }
        }

        // ---- helpers -------------------------------------------------------------------------------------------

        static ReferenceSolver Colored(string scene, out AvbdSolverSettings settings)
        {
            var solver = new ReferenceSolver();
            switch (scene)
            {
                case "ground": AvbdScenes.Ground(solver); break;
                case "stack": AvbdScenes.Stack(solver); break;
                case "overlap": // a box sunk into the ground and a second one sunk into it, both off-center: several contacts per pair
                    solver.Clear();
                    solver.AddBody(new float3(100f, 100f, 1f), 0f, 0.5f, new float3(0f, 0f, 0f));
                    solver.AddBody(new float3(1f, 1f, 1f), 1f, 0.5f, new float3(0.3f, 0.2f, 0.95f));
                    solver.AddBody(new float3(1f, 1f, 1f), 1f, 0.5f, new float3(0.6f, 0.1f, 1.9f));
                    break;
                default: throw new ArgumentException("Unknown scene " + scene);
            }

            settings = AvbdSolverSettings.Default;
            settings.gravity = new float3(0f, 0f, -10f); // the upstream scenes are z-up
            solver.Params.dt = settings.dt;
            solver.Params.gravity = settings.gravity;
            solver.Params.iterations = settings.iterations;
            solver.Params.mode = GaussSeidelMode.Colored;
            return solver;
        }

        // The oracle for a converted scene: a record array with room for the contacts of all its bodies.
        static CpuContacts Oracle(SceneArrays arrays, out ConstraintRecord[] records)
        {
            records = new ConstraintRecord[CpuContacts.RecordCapacity(arrays.constraintCount, arrays.bodyCount)];
            Array.Copy(arrays.constraints, records, arrays.constraintCount);
            var contacts = new CpuContacts(arrays.bodyCount, arrays.constraintCount);
            contacts.ExcludeJoinedPairs(records, arrays.constraintCount);
            return contacts;
        }

        static bool HasContacts(CpuContacts contacts)
        {
            foreach (uint n in contacts.PairCounts)
                if (n > 0u)
                    return true;
            return false;
        }

        static int[] Checkpoints(string scene) => scene == "ground" ? new[] { 1, 10, 30, 60, 90 } : new[] { 1, 2, 10, 30, 60 };

        static float PositionTolerance(string scene) => scene == "ground" ? 5e-3f : 0.1f;

        static float RotationTolerance(string scene) => scene == "ground" ? 1e-3f : 1e-2f;

        static float VelocityTolerance(string scene) => scene == "ground" ? 5e-2f : 0.5f; // relative to max(1, |v|)

        static void AssertBodyNear(AvbdBody expected, CpuBodyArrays actual, int slot, string scene, int step)
        {
            string what = scene + " step " + step + " slot " + slot;
            AssertNear(expected.PositionLin, actual.State[slot].position.xyz, PositionTolerance(scene), what + " position");
            AssertNear(expected.PositionAng.value, actual.State[slot].rotation, RotationTolerance(scene), what + " rotation");
            AssertNear(expected.VelocityLin, actual.Motion[slot].linearVel.xyz, VelocityTolerance(scene), what + " linear velocity");
            AssertNear(expected.VelocityAng, actual.Motion[slot].angularVel.xyz, VelocityTolerance(scene), what + " angular velocity");
        }

        // Every manifold of the reference solver is a pair of the oracle with the same contacts, and the oracle has no other pairs.
        static void AssertManifoldsMatch(ReferenceSolver solver, CpuContacts contacts, ConstraintRecord[] records, string scene, float tolerance)
        {
            var expectedPairs = new HashSet<int>();
            int manifolds = 0;
            foreach (AvbdForce force in solver.Forces)
            {
                if (!(force is AvbdManifold m))
                    continue;
                manifolds++;
                int a = m.BodyA.Index;
                int b = m.BodyB.Index;
                Assert.Greater(a, b, "the reference solver's body A is the higher slot");
                int pair = (int)ContactLayout.PairIndex(a, b);
                expectedPairs.Add(pair);

                Assert.AreEqual(m.NumContacts, (int)contacts.PairCounts[pair], scene + " pair (" + a + ", " + b + ") contact count");
                for (int i = 0; i < m.NumContacts; i++)
                {
                    AvbdManifold.Contact c = m.Contacts[i];
                    ConstraintRecord r = records[contacts.ContactBase + pair * ContactLayout.MaxContactsPerPair + i];
                    string what = scene + " pair (" + a + ", " + b + ") contact " + i;

                    Assert.AreEqual((uint)a, r.ident.x, what + " body A");
                    Assert.AreEqual((uint)b, r.ident.y, what + " body B");
                    Assert.AreEqual((uint)ConstraintKind.Contact, r.ident.z >> ConstraintFlagLayout.KindShift, what + " kind");
                    Assert.AreEqual(c.featureKey, (int)r.ident.w, what + " feature key");
                    AssertNear(c.rA, r.anchorA.xyz, tolerance, what + " rA");
                    AssertNear(c.rB, r.anchorB.xyz, tolerance, what + " rB");
                    AssertNear(c.C0, r.c0Lin.xyz, tolerance, what + " C0");
                    AssertNear(AvbdMat3.Row(m.Basis, 0), r.c0Ang.xyz, tolerance, what + " normal");
                    Assert.AreEqual(m.Friction, r.anchorA.w, tolerance, what + " friction");
                }
            }
            Assert.Greater(manifolds, 0, scene + ": the reference solver found no manifold, so nothing was tested");

            for (int pair = 0; pair < contacts.PairCounts.Length; pair++)
                if (contacts.PairCounts[pair] > 0u)
                    Assert.IsTrue(expectedPairs.Contains(pair), scene + ": the oracle has contacts at pair " + pair + " that the reference solver does not");
        }

        // Records of the same contact: identity exactly, values within a tolerance relative to max(1, |value|).
        static void AssertRecordNear(ConstraintRecord expected, ConstraintRecord actual, float tol, string what)
        {
            Assert.AreEqual(expected.ident, actual.ident, what + " ident");
            AssertNear(expected.anchorA, actual.anchorA, tol, what + " anchorA");
            AssertNear(expected.anchorB, actual.anchorB, tol, what + " anchorB");
            AssertNear(expected.c0Lin, actual.c0Lin, tol, what + " C0");
            AssertNear(expected.c0Ang, actual.c0Ang, tol, what + " normal");
            AssertNear(expected.penaltyLin, actual.penaltyLin, tol, what + " penalty");
            AssertNear(expected.lambdaLin, actual.lambdaLin, tol, what + " lambda");
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

        static void AssertNear(float3 expected, float3 actual, float tol, string what)
        {
            for (int k = 0; k < 3; k++)
                Assert.AreEqual(expected[k], actual[k], tol * math.max(1f, math.abs(expected[k])), what + "[" + k + "]");
        }

        static void AssertNear(float4 expected, float4 actual, float tol, string what)
        {
            for (int k = 0; k < 4; k++)
                Assert.AreEqual(expected[k], actual[k], tol * math.max(1f, math.abs(expected[k])), what + "[" + k + "]");
        }
    }
}
