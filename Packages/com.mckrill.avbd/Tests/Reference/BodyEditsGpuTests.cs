using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Runs stage E on the GPU and compares the arrays with the CPU oracle bit for bit.
    /// Needs a GPU with compute shader support; the tests are ignored without one.
    /// </summary>
    public class BodyEditsGpuTests
    {
        const int Capacity = 256;

        [SetUp]
        public void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

        // ---- simple, readable cases ---------------------------------------------------------------------

        [Test]
        public void NewBuffers_AreClearedToTheFreeState()
        {
            using (var buffers = new AvbdBodyBuffers(Capacity))
                AssertMatches(buffers, new CpuBodyArrays(Capacity), "after creation");
        }

        [Test]
        public void ApplyEdits_WithNothingPending_DoesNothing()
        {
            using (var buffers = new AvbdBodyBuffers(16))
            {
                Assert.IsFalse(buffers.HasPendingEdits);
                Assert.AreEqual(0, buffers.ApplyEdits());
            }
        }

        [Test]
        public void Spawn_IsVisibleOnTheGpuAfterApply()
        {
            using (var buffers = new AvbdBodyBuffers(16))
            {
                BodyDescriptor d = BodyDescriptor.Box(new float3(1f, 2f, 3f), 2f, new float3(4f, 5f, 6f));
                d.appearanceId = 11u;
                BodyHandle h = buffers.Spawn(d);
                Assert.IsTrue(buffers.HasPendingEdits);
                Assert.AreEqual(1, buffers.ApplyEdits());

                var state = new BodyStateRecord[16];
                var meta = new BodyMetaRecord[16];
                var mass = new BodyMassRecord[16];
                buffers.State.GetData(state);
                buffers.Meta.GetData(meta);
                buffers.Mass.GetData(mass);

                int s = h.Slot;
                Assert.AreEqual(4f, state[s].position.x);
                Assert.AreEqual(5f, state[s].position.y);
                Assert.AreEqual(6f, state[s].position.z);
                Assert.AreEqual(1f, state[s].rotation.w);
                Assert.AreEqual(0u, meta[s].ident.x & (uint)BodyFlags.Free, "the slot must be live");
                Assert.AreEqual(11u, meta[s].ident.z);
                Assert.AreEqual(12f, mass[s].momentMass.w, 1e-5f);
            }
        }

        [Test]
        public void Remove_MarksTheSlotFreeOnTheGpu()
        {
            using (var buffers = new AvbdBodyBuffers(16))
            {
                BodyHandle h = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
                buffers.ApplyEdits();
                Assert.IsTrue(buffers.Remove(h));
                buffers.ApplyEdits();

                var meta = new BodyMetaRecord[16];
                var mass = new BodyMassRecord[16];
                buffers.Meta.GetData(meta);
                buffers.Mass.GetData(mass);

                Assert.AreNotEqual(0u, meta[h.Slot].ident.x & (uint)BodyFlags.Free);
                Assert.AreEqual(AvbdConst.Invalid, meta[h.Slot].ident.z);
                Assert.AreEqual(0f, mass[h.Slot].momentMass.w);
            }
        }

        [Test]
        public void Spawn_WhenFull_ReturnsInvalid()
        {
            using (var buffers = new AvbdBodyBuffers(2))
            {
                buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
                buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));
                BodyHandle none = buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero));

                Assert.IsFalse(none.IsValid);
                Assert.AreEqual(1, buffers.OverflowCount);
                Assert.AreEqual(2, buffers.AliveCount);
            }
        }

        [Test]
        public void DisposedBuffers_RejectFurtherUse()
        {
            var buffers = new AvbdBodyBuffers(4);
            buffers.Dispose();
            Assert.Throws<ObjectDisposedException>(() => buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero)));
            Assert.DoesNotThrow(buffers.Dispose, "Dispose must be idempotent");
        }

        [Test]
        public void Constructor_RejectsBadArguments()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new AvbdBodyBuffers(0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new AvbdBodyBuffers(4, 0));
        }

        // ---- the main check: random edit streams against the oracle ----------------------------------------

        [Test]
        public void RandomEdits_MatchTheCpuOracleBitForBit()
        {
            var rng = new System.Random(20260611);
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();
            var live = new List<BodyHandle>();

            // A small command batch forces the upload to be split over several dispatches.
            using (var buffers = new AvbdBodyBuffers(Capacity, commandBatchSize: 64))
            {
                AssertMatches(buffers, oracle, "after creation");

                for (int batch = 0; batch < 40; batch++)
                {
                    int ops = rng.Next(0, 160);
                    for (int i = 0; i < ops; i++)
                        RandomOp(rng, buffers, live);

                    applied.Clear();
                    buffers.ApplyEdits(applied);
                    oracle.Apply(applied);

                    AssertMatches(buffers, oracle, "after batch " + batch);
                    Assert.AreEqual(live.Count, buffers.AliveCount, "alive count after batch " + batch);
                }
            }
        }

        [Test]
        public void ManySpawnsAndRemoves_ThatFillAndEmptyTheBuffers_MatchTheOracle()
        {
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();
            var handles = new List<BodyHandle>();

            using (var buffers = new AvbdBodyBuffers(Capacity, commandBatchSize: 100))
            {
                for (int round = 0; round < 3; round++)
                {
                    for (int i = 0; i < Capacity; i++)
                    {
                        BodyDescriptor d = BodyDescriptor.Box(new float3(1f + i * 0.01f), 1f, new float3(i, round, 0f));
                        d.appearanceId = (uint)i;
                        BodyHandle h = buffers.Spawn(d);
                        Assert.IsTrue(h.IsValid);
                        handles.Add(h);
                    }
                    Assert.IsFalse(buffers.Spawn(BodyDescriptor.Box(new float3(1f), 1f, float3.zero)).IsValid, "buffers must be full");

                    applied.Clear();
                    buffers.ApplyEdits(applied);
                    oracle.Apply(applied);
                    AssertMatches(buffers, oracle, "full, round " + round);

                    foreach (BodyHandle h in handles)
                        Assert.IsTrue(buffers.Remove(h));
                    handles.Clear();

                    applied.Clear();
                    buffers.ApplyEdits(applied);
                    oracle.Apply(applied);
                    AssertMatches(buffers, oracle, "empty, round " + round);
                }
            }
        }

        // ---- random edit generation -----------------------------------------------------------------------

        static float F(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static float3 V(System.Random rng, float range) => new float3(F(rng, -range, range), F(rng, -range, range), F(rng, -range, range));

        static quaternion Q(System.Random rng) => quaternion.EulerXYZ(F(rng, 0f, 6.28f), F(rng, 0f, 6.28f), F(rng, 0f, 6.28f));

        static void RandomOp(System.Random rng, AvbdBodyBuffers buffers, List<BodyHandle> live)
        {
            int pick = rng.Next(100);

            if (pick < 40 || live.Count == 0)
            {
                BodyDescriptor d = BodyDescriptor.Box(new float3(F(rng, 0.2f, 3f), F(rng, 0.2f, 3f), F(rng, 0.2f, 3f)),
                    rng.Next(5) == 0 ? 0f : F(rng, 0.1f, 2f), V(rng, 50f));
                d.rotation = Q(rng);
                d.linearVelocity = V(rng, 5f);
                d.angularVelocity = V(rng, 5f);
                d.friction = F(rng, 0f, 1f);
                d.groupMask = (uint)rng.Next();
                d.appearanceId = (uint)rng.Next(1000);
                d.kinematic = rng.Next(4) == 0;
                BodyHandle h = buffers.Spawn(d);
                if (h.IsValid)
                    live.Add(h);
                return;
            }

            int index = rng.Next(live.Count);
            BodyHandle target = live[index];

            if (pick < 55)
            {
                Assert.IsTrue(buffers.Remove(target));
                live.RemoveAt(index);
                return;
            }

            switch (rng.Next(7))
            {
                case 0: Assert.IsTrue(buffers.SetPose(target, V(rng, 50f), Q(rng))); break;
                case 1: Assert.IsTrue(buffers.SetVelocity(target, V(rng, 5f), V(rng, 5f))); break;
                case 2: Assert.IsTrue(buffers.SetShape(target, new float3(F(rng, 0.2f, 3f), F(rng, 0.2f, 3f), F(rng, 0.2f, 3f)), rng.Next(5) == 0 ? 0f : F(rng, 0.1f, 2f))); break;
                case 3: Assert.IsTrue(buffers.SetFriction(target, F(rng, 0f, 1f))); break;
                case 4: Assert.IsTrue(buffers.SetGroupMask(target, (uint)rng.Next())); break;
                case 5: Assert.IsTrue(buffers.SetAppearanceId(target, (uint)rng.Next(1000))); break;
                default: Assert.IsTrue(buffers.SetKinematic(target, rng.Next(2) == 0)); break;
            }
        }

        // ---- comparison -----------------------------------------------------------------------------------

        static void AssertMatches(AvbdBodyBuffers buffers, CpuBodyArrays oracle, string when)
        {
            AssertBuffer(buffers.State, oracle.State, "State", when);
            AssertBuffer(buffers.Inertial, oracle.Inertial, "Inertial", when);
            AssertBuffer(buffers.Mass, oracle.Mass, "Mass", when);
            AssertBuffer(buffers.Motion, oracle.Motion, "Motion", when);
            AssertBuffer(buffers.Shape, oracle.Shape, "Shape", when);
            AssertBuffer(buffers.Meta, oracle.Meta, "Meta", when);
            AssertBuffer(buffers.Range, oracle.Range, "Range", when);
        }

        // Synchronous readback is fine in a test. The simulation never reads back.
        static void AssertBuffer<T>(GraphicsBuffer buffer, T[] expected, string name, string when) where T : struct
        {
            var actual = new T[expected.Length];
            buffer.GetData(actual);

            for (int i = 0; i < expected.Length; i++)
            {
                ReadOnlySpan<byte> e = MemoryMarshal.AsBytes(new ReadOnlySpan<T>(expected, i, 1));
                ReadOnlySpan<byte> a = MemoryMarshal.AsBytes(new ReadOnlySpan<T>(actual, i, 1));
                if (!e.SequenceEqual(a))
                    Assert.Fail(name + "[" + i + "] differs " + when + ".\n  expected: " + expected[i] + "\n  GPU:      " + actual[i]);
            }
        }
    }
}
