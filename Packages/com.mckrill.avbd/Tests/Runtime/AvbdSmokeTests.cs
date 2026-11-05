using NUnit.Framework;
using UnityEngine;

namespace McKrill.Avbd.Tests
{
    public class AvbdSmokeTests
    {
        const int Count = 1000;
        const int GroupSize = 64;

        [Test]
        public void ShaderSet_LoadsFromResources()
        {
            var set = AvbdShaderSet.Load();

            Assert.IsNotNull(set, "AvbdShaderSet asset was not found in a Resources folder.");
            Assert.IsNotNull(set.Smoke, "AvbdShaderSet has no smoke compute shader assigned.");
        }

        [Test]
        public void SmokeKernel_WritesExpectedPattern()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");

            var shader = AvbdShaderSet.Load().Smoke;
            int kernel = shader.FindKernel("CSMain");

            using (var buffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Count, sizeof(uint)))
            {
                shader.SetBuffer(kernel, "_Output", buffer);
                shader.SetInt("_Count", Count);
                shader.Dispatch(kernel, (Count + GroupSize - 1) / GroupSize, 1, 1);

                // Synchronous readback is fine in a test. The simulation hot path never reads back.
                var result = new uint[Count];
                buffer.GetData(result);

                for (uint i = 0; i < Count; i++)
                    Assert.AreEqual(i * 3u + 7u, result[i], $"Mismatch at index {i}.");
            }
        }
    }
}
