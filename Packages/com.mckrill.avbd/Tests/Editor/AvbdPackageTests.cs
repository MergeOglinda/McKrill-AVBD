using NUnit.Framework;
using UnityEditor.PackageManager;

namespace McKrill.Avbd.Tests
{
    public class AvbdPackageTests
    {
        [Test]
        public void PackageManifest_HasExpectedIdentity()
        {
            var info = PackageInfo.FindForAssembly(typeof(AvbdShaderSet).Assembly);

            Assert.IsNotNull(info, "The runtime assembly is not part of a package.");
            Assert.AreEqual("com.mckrill.avbd", info.name);
            Assert.AreEqual("McKrill AVBD", info.displayName);
        }

        [Test]
        public void PackageManifest_VersionIsSemantic()
        {
            var info = PackageInfo.FindForAssembly(typeof(AvbdShaderSet).Assembly);

            Assert.IsNotNull(info);
            StringAssert.IsMatch(@"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$", info.version);
        }
    }
}
