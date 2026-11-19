using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Tests
{
    /// <summary>The GPU ray cast (<see cref="AvbdRaycaster"/>) against boxes whose hit distances are known by hand.</summary>
    public class RaycastTests
    {
        const float Max = 1000f;

        [Test]
        public void Cast_FindsTheNearestBox_AndIgnoresBoxesBehindAndBesideTheRay()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var raycaster = new AvbdRaycaster(buffers))
            {
                buffers.Spawn(BodyDescriptor.Box(new float3(2f), 1f, new float3(0f, 0f, 10f)));   // front face at z = 9
                buffers.Spawn(BodyDescriptor.Box(new float3(2f), 1f, new float3(0f, 0f, 5f)));    // front face at z = 4: the nearest
                buffers.Spawn(BodyDescriptor.Box(new float3(2f), 1f, new float3(4f, 0f, 3f)));    // beside the ray
                buffers.Spawn(BodyDescriptor.Box(new float3(2f), 1f, new float3(0f, 0f, -5f)));   // behind the origin

                Assert.AreEqual(4f, Cast(buffers, raycaster, new float3(0f), new float3(0f, 0f, 1f), Max), 1e-4f);
            }
        }

        [Test]
        public void Cast_UsesTheRotationAndTheSizeOfTheBox_AndNormalizesTheDirection()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var raycaster = new AvbdRaycaster(buffers))
            {
                BodyDescriptor turned = BodyDescriptor.Box(new float3(1f, 1f, 4f), 1f, new float3(0f, 0f, 10f));
                turned.rotation = quaternion.RotateY(math.radians(90f)); // the 4 m side now lies along x: the box is 1 m deep in z
                buffers.Spawn(turned);

                // A direction of length 3 gives the same distance as a unit one: distances are in meters.
                Assert.AreEqual(9.5f, Cast(buffers, raycaster, new float3(0f), new float3(0f, 0f, 3f), Max), 1e-4f);
                // Along x at the box's height, the long side is 4 m wide, so the ray at z = 10 hits the face at x = -2.
                Assert.AreEqual(2f, Cast(buffers, raycaster, new float3(-4f, 0f, 10f), new float3(1f, 0f, 0f), Max), 1e-4f);
            }
        }

        [Test]
        public void Cast_Misses_WhenTheRayPassesBesideTheBox_OrTheBoxIsFartherThanTheMaximum_OrTheBodyWasRemoved()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var raycaster = new AvbdRaycaster(buffers))
            {
                BodyHandle h = buffers.Spawn(BodyDescriptor.Box(new float3(2f), 1f, new float3(0f, 0f, 5f)));

                Assert.IsTrue(float.IsNaN(Cast(buffers, raycaster, new float3(1.5f, 0f, 0f), new float3(0f, 1f, 0f), Max)), "parallel to a face, beside the box");
                Assert.IsTrue(float.IsNaN(Cast(buffers, raycaster, new float3(0f), new float3(0f, 0f, 1f), 3f)), "the box starts at 4 m and the ray reaches 3");
                Assert.AreEqual(4f, Cast(buffers, raycaster, new float3(0f), new float3(0f, 0f, 1f), Max), 1e-4f);

                Assert.IsTrue(buffers.Remove(h));
                Assert.IsTrue(float.IsNaN(Cast(buffers, raycaster, new float3(0f), new float3(0f, 0f, 1f), Max)), "a removed body is not hit");
            }
        }

        [Test]
        public void Cast_FromInsideABox_HitsItAtDistanceZero()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var raycaster = new AvbdRaycaster(buffers))
            {
                buffers.Spawn(BodyDescriptor.Box(new float3(4f), 0f, float3.zero)); // a static body
                Assert.AreEqual(0f, Cast(buffers, raycaster, new float3(0.5f, 0.25f, 0f), new float3(1f, 0f, 0f), Max), 0f);
            }
        }

        [Test]
        public void Cast_RejectsBadRays()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var raycaster = new AvbdRaycaster(buffers))
            {
                Assert.Throws<System.ArgumentException>(() => raycaster.Cast(float3.zero, float3.zero, Max, (h, d) => { }));
                Assert.Throws<System.ArgumentException>(() => raycaster.Cast(new float3(float.NaN, 0f, 0f), new float3(1f, 0f, 0f), Max, (h, d) => { }));
                Assert.Throws<System.ArgumentException>(() => raycaster.Cast(float3.zero, new float3(1f, 0f, 0f), 0f, (h, d) => { }));
                Assert.Throws<System.ArgumentNullException>(() => raycaster.Cast(float3.zero, new float3(1f, 0f, 0f), Max, null));
            }
        }

        // ---- helpers --------------------------------------------------------------------------------------------

        // Runs the cast and reads the one-word result back at once (the tests do not wait for frames). NaN means a miss.
        static float Cast(AvbdBodyBuffers buffers, AvbdRaycaster raycaster, float3 origin, float3 direction, float max)
        {
            raycaster.DispatchCast(origin, direction, max);
            var bits = new uint[1];
            raycaster.ResultBuffer.GetData(bits);
            return bits[0] == math.asuint(max) ? float.NaN : math.asfloat(bits[0]);
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }
    }
}
