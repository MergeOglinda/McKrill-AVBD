using System;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Tests
{
    /// <summary>The GPU group labeler (<see cref="AvbdGroupLabeler"/>) on boxes whose distances to a point are known by hand.</summary>
    public class GroupLabelerTests
    {
        [Test]
        public void Label_SetsTheBitOfTheBodiesWithinTheRadius_AndOnlyThose()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var labeler = new AvbdGroupLabeler(buffers))
            {
                BodyHandle near = Spawn(buffers, new float3(0f), 1u);               // 2 m cube, x in [-1, 1]: 0.5 m from the point (1.5, 0, 0)
                BodyHandle alsoNear = Spawn(buffers, new float3(3f, 0f, 0f), 1u);   // x in [2, 4]: 0.5 m from the point
                BodyHandle far = Spawn(buffers, new float3(10f, 0f, 0f), 1u);       // 7.5 m

                labeler.Label(new float3(1.5f, 0f, 0f), 0.6f, 4u, 0u);
                Assert.AreEqual(5u, Mask(buffers, near));
                Assert.AreEqual(5u, Mask(buffers, alsoNear));
                Assert.AreEqual(1u, Mask(buffers, far));

                labeler.Label(new float3(1.5f, 0f, 0f), 0.4f, 8u, 0u);
                Assert.AreEqual(5u, Mask(buffers, near), "a radius smaller than the distance selects nothing");
            }
        }

        [Test]
        public void Label_MeasuresTheDistanceToTheBox_NotToItsCenter_AndUsesItsRotation()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var labeler = new AvbdGroupLabeler(buffers))
            {
                // A 1 x 1 x 6 box along z, turned 90 degrees about y: it now lies along x, from x = -3 to 3, 0.5 m thick.
                BodyDescriptor beam = BodyDescriptor.Box(new float3(1f, 1f, 6f), 1f, new float3(0f));
                beam.rotation = quaternion.RotateY(math.radians(90f));
                beam.groupMask = 0u;
                BodyHandle h = buffers.Spawn(beam);

                // The point at x = 2.9 is inside the beam's length but 2.9 m from its center, and 0.5 m from the surface.
                labeler.Label(new float3(2.9f, 1f, 0f), 0.4f, 2u, 0u);
                Assert.AreEqual(0u, Mask(buffers, h), "0.5 m from the surface, radius 0.4");
                labeler.Label(new float3(2.9f, 1f, 0f), 0.6f, 2u, 0u);
                Assert.AreEqual(2u, Mask(buffers, h), "the long side counts: the center is 2.9 m away");

                // Inside the box the distance is zero, so a radius of zero selects it.
                labeler.Label(new float3(-2.5f, 0.2f, 0.2f), 0f, 16u, 0u);
                Assert.AreEqual(18u, Mask(buffers, h));
            }
        }

        [Test]
        public void Label_ClearsBits_AndClearAllReachesEveryBody()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var labeler = new AvbdGroupLabeler(buffers))
            {
                BodyHandle a = Spawn(buffers, new float3(0f), 3u);
                BodyHandle b = Spawn(buffers, new float3(500f, 0f, 0f), 3u);

                labeler.Label(new float3(0f), 0.1f, 4u, 1u); // sets 4 and clears 1 on the body at the point
                Assert.AreEqual(6u, Mask(buffers, a));
                Assert.AreEqual(3u, Mask(buffers, b));

                labeler.ClearAll(2u);
                Assert.AreEqual(4u, Mask(buffers, a));
                Assert.AreEqual(1u, Mask(buffers, b), "ClearAll also reaches the body 500 m away");
            }
        }

        [Test]
        public void Label_RejectsBadValues()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var labeler = new AvbdGroupLabeler(buffers))
            {
                Assert.Throws<ArgumentException>(() => labeler.Label(new float3(float.NaN, 0f, 0f), 1f, 1u, 0u));
                Assert.Throws<ArgumentException>(() => labeler.Label(float3.zero, -1f, 1u, 0u));
                Assert.Throws<ArgumentException>(() => labeler.Label(float3.zero, float.PositiveInfinity, 1u, 0u));
                labeler.Dispose();
                Assert.Throws<ObjectDisposedException>(() => labeler.Label(float3.zero, 1f, 1u, 0u));
            }
        }

        // A 2 m cube.
        static BodyHandle Spawn(AvbdBodyBuffers buffers, float3 position, uint groupMask)
        {
            BodyDescriptor d = BodyDescriptor.Box(new float3(2f), 1f, position);
            d.groupMask = groupMask;
            return buffers.Spawn(d);
        }

        static uint Mask(AvbdBodyBuffers buffers, BodyHandle handle)
        {
            var meta = new BodyMetaRecord[buffers.Capacity];
            buffers.Meta.GetData(meta);
            return meta[handle.Slot].ident.y;
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
