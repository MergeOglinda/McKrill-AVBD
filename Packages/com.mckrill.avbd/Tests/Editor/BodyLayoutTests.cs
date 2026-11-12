using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Text.RegularExpressions;
using NUnit.Framework;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEditor.PackageManager;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Checks the C# record mirrors against the packing rules and against the HLSL includes they mirror.
    /// These run on the CPU only, so a layout drift between C# and HLSL fails here without needing a GPU.
    /// </summary>
    public class BodyLayoutTests
    {
        // ---- sizes ----------------------------------------------------------------------------------

        [Test]
        public void RecordStrides_MatchTheStructSizes()
        {
            Assert.AreEqual(BodyRecordStrides.State, UnsafeUtility.SizeOf<BodyStateRecord>());
            Assert.AreEqual(BodyRecordStrides.Inertial, UnsafeUtility.SizeOf<BodyInertialRecord>());
            Assert.AreEqual(BodyRecordStrides.Mass, UnsafeUtility.SizeOf<BodyMassRecord>());
            Assert.AreEqual(BodyRecordStrides.Motion, UnsafeUtility.SizeOf<BodyMotionRecord>());
            Assert.AreEqual(BodyRecordStrides.Shape, UnsafeUtility.SizeOf<BodyShapeRecord>());
            Assert.AreEqual(BodyRecordStrides.Meta, UnsafeUtility.SizeOf<BodyMetaRecord>());
            Assert.AreEqual(BodyRecordStrides.Range, UnsafeUtility.SizeOf<BodyRangeRecord>());
            Assert.AreEqual(EditCommandStride.Size, UnsafeUtility.SizeOf<EditCommand>());
        }

        [Test]
        public void RecordStrides_FollowRuleP3()
        {
            int[] all =
            {
                BodyRecordStrides.State, BodyRecordStrides.Inertial, BodyRecordStrides.Mass, BodyRecordStrides.Motion,
                BodyRecordStrides.Shape, BodyRecordStrides.Meta, BodyRecordStrides.Range, EditCommandStride.Size,
            };
            foreach (int stride in all)
            {
                Assert.AreEqual(0, stride % 16, "stride " + stride + " is not a multiple of 16");
                Assert.Less(stride, 2048);
            }

            // Records the solver stages gather by body id: 16 B or a multiple of 32 B.
            int[] gathered = { BodyRecordStrides.State, BodyRecordStrides.Inertial, BodyRecordStrides.Mass, BodyRecordStrides.Shape, BodyRecordStrides.Range };
            foreach (int stride in gathered)
                Assert.IsTrue(stride == 16 || stride % 32 == 0, "gathered stride " + stride + " breaks rule P3");
        }

        [Test]
        public void Total_IsTheSumOfTheArrays()
        {
            Assert.AreEqual(224, BodyRecordStrides.Total);
            Assert.AreEqual(BodyRecordStrides.Total, BodyBufferCapacity.BytesPerBody);
            Assert.AreEqual(224000L, BodyBufferCapacity.BodyArrayBytes(1000));
            Assert.AreEqual(BodyRecordStrides.Largest, Math.Max(
                Math.Max(BodyRecordStrides.State, BodyRecordStrides.Inertial),
                Math.Max(Math.Max(BodyRecordStrides.Mass, BodyRecordStrides.Motion),
                    Math.Max(BodyRecordStrides.Shape, Math.Max(BodyRecordStrides.Meta, BodyRecordStrides.Range)))));
        }

        // ---- capacity (rule P6) ---------------------------------------------------------------------

        [Test]
        public void Capacity_LimitsAreEnforced()
        {
            Assert.IsFalse(BodyBufferCapacity.TryValidate(0, out _));
            Assert.IsFalse(BodyBufferCapacity.TryValidate(-5, out _));
            Assert.IsTrue(BodyBufferCapacity.TryValidate(1, out _));
            Assert.IsTrue(BodyBufferCapacity.TryValidate(BodyBufferCapacity.MaxBodies, out _));
            Assert.IsFalse(BodyBufferCapacity.TryValidate(BodyBufferCapacity.MaxBodies + 1, out string error));
            StringAssert.Contains("maximum", error);
            Assert.Throws<ArgumentOutOfRangeException>(() => BodyBufferCapacity.Validate(0));
        }

        [Test]
        public void Capacity_KeepsEveryIdBelowTheTagBit()
        {
            Assert.Less((long)BodyBufferCapacity.MaxBodies, 0x7FFFFFFFL);
            Assert.LessOrEqual((long)BodyBufferCapacity.MaxBodies * BodyRecordStrides.Largest, (long)int.MaxValue);
            Assert.AreEqual(0xFFFFFFFFu, AvbdConst.Invalid);
        }

        // ---- flags (rule P7) --------------------------------------------------------------------------

        [Test]
        public void Flags_AreDistinctSingleBitsInTheLow24()
        {
            BodyFlags[] flags = { BodyFlags.Free, BodyFlags.Kinematic, BodyFlags.Asleep };
            uint seen = 0;
            foreach (BodyFlags flag in flags)
            {
                uint bit = (uint)flag;
                Assert.AreEqual(1, math.countbits(bit), flag + " is not a single bit");
                Assert.AreEqual(0u, bit & ~BodyFlagLayout.FlagsMask, flag + " is outside the low 24 bits");
                Assert.AreEqual(0u, seen & bit, flag + " overlaps another flag");
                seen |= bit;
            }
        }

        [Test]
        public void Flags_PackAndUnpack()
        {
            uint packed = BodyFlagLayout.Pack(BodyFlags.Kinematic | BodyFlags.Asleep, ShapeKind.Box);
            Assert.AreEqual(BodyFlags.Kinematic | BodyFlags.Asleep, BodyFlagLayout.FlagsOf(packed));
            Assert.AreEqual(ShapeKind.Box, BodyFlagLayout.KindOf(packed));
            Assert.AreEqual(0u, packed >> BodyFlagLayout.KindShift);
        }

        [Test]
        public void StageE_MayOnlySetTheKinematicBit()
        {
            Assert.AreEqual((uint)BodyFlags.Kinematic, BodyFlagLayout.UserSettable);
            Assert.AreEqual(0u, BodyFlagLayout.UserSettable & (uint)BodyFlags.Free);
            Assert.AreEqual(0u, BodyFlagLayout.UserSettable & (uint)BodyFlags.Asleep);
        }

        // ---- C# mirrors against the HLSL includes -----------------------------------------------------

        [Test]
        public void Hlsl_ConstantsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdCommon.hlsl", "AvbdBodyRecords.hlsl", "AvbdEditCommand.hlsl");

            Assert.AreEqual(AvbdConst.Invalid, d["AVBD_INVALID"]);
            Assert.AreEqual((uint)AvbdConst.GroupSize, d["AVBD_GROUP_SIZE"]);

            Assert.AreEqual((uint)BodyFlags.Free, d["AVBD_BODY_FLAG_FREE"]);
            Assert.AreEqual((uint)BodyFlags.Kinematic, d["AVBD_BODY_FLAG_KINEMATIC"]);
            Assert.AreEqual((uint)BodyFlags.Asleep, d["AVBD_BODY_FLAG_ASLEEP"]);
            Assert.AreEqual(BodyFlagLayout.FlagsMask, d["AVBD_BODY_FLAGS_MASK"]);
            Assert.AreEqual((uint)BodyFlagLayout.KindShift, d["AVBD_BODY_KIND_SHIFT"]);
            Assert.AreEqual((uint)ShapeKind.Box, d["AVBD_KIND_BOX"]);

            Assert.AreEqual((uint)EditKind.Spawn, d["AVBD_EDIT_SPAWN"]);
            Assert.AreEqual((uint)EditKind.Remove, d["AVBD_EDIT_REMOVE"]);
            Assert.AreEqual((uint)EditKind.Set, d["AVBD_EDIT_SET"]);

            Assert.AreEqual((uint)EditFields.Pose, d["AVBD_EDIT_FIELD_POSE"]);
            Assert.AreEqual((uint)EditFields.Velocity, d["AVBD_EDIT_FIELD_VELOCITY"]);
            Assert.AreEqual((uint)EditFields.Shape, d["AVBD_EDIT_FIELD_SHAPE"]);
            Assert.AreEqual((uint)EditFields.Friction, d["AVBD_EDIT_FIELD_FRICTION"]);
            Assert.AreEqual((uint)EditFields.GroupMask, d["AVBD_EDIT_FIELD_GROUPMASK"]);
            Assert.AreEqual((uint)EditFields.Appearance, d["AVBD_EDIT_FIELD_APPEARANCE"]);
            Assert.AreEqual((uint)EditFields.Kinematic, d["AVBD_EDIT_FIELD_KINEMATIC"]);
        }

        [Test]
        public void Hlsl_EditFieldsAllIsTheUnionOfTheFields()
        {
            uint union = 0;
            foreach (EditFields field in Enum.GetValues(typeof(EditFields)))
                if (field != EditFields.All)
                    union |= (uint)field;
            Assert.AreEqual((uint)EditFields.All, union);
        }

        [Test]
        public void Hlsl_StructsMatchTheCSharpRecords()
        {
            string records = ReadInclude("AvbdBodyRecords.hlsl");
            string commands = ReadInclude("AvbdEditCommand.hlsl");

            AssertStructMatches(records, "BodyStateRecord", typeof(BodyStateRecord));
            AssertStructMatches(records, "BodyInertialRecord", typeof(BodyInertialRecord));
            AssertStructMatches(records, "BodyMassRecord", typeof(BodyMassRecord));
            AssertStructMatches(records, "BodyMotionRecord", typeof(BodyMotionRecord));
            AssertStructMatches(records, "BodyShapeRecord", typeof(BodyShapeRecord));
            AssertStructMatches(records, "BodyMetaRecord", typeof(BodyMetaRecord));
            AssertStructMatches(records, "BodyRangeRecord", typeof(BodyRangeRecord));
            AssertStructMatches(commands, "EditCommand", typeof(EditCommand));
        }

        static void AssertStructMatches(string hlslSource, string hlslName, Type csharpType)
        {
            Match match = Regex.Match(hlslSource, @"struct\s+" + hlslName + @"\s*\{([^}]*)\}");
            Assert.IsTrue(match.Success, "struct " + hlslName + " not found in HLSL");

            var hlslMembers = new List<string>();
            foreach (string rawLine in match.Groups[1].Value.Split('\n'))
            {
                string line = StripComment(rawLine).Trim();
                if (line.Length == 0)
                    continue;
                Match member = Regex.Match(line, @"^(\w+)\s+(\w+)\s*;$");
                Assert.IsTrue(member.Success, hlslName + ": cannot parse member '" + line + "'");
                Assert.IsTrue(member.Groups[1].Value == "float4" || member.Groups[1].Value == "uint4",
                    hlslName + "." + member.Groups[2].Value + " is " + member.Groups[1].Value + " (rule P1 allows only float4 and uint4)");
                hlslMembers.Add(member.Groups[1].Value + " " + member.Groups[2].Value);
            }

            var csharpMembers = new List<string>();
            foreach (FieldInfo field in csharpType.GetFields(BindingFlags.Public | BindingFlags.Instance))
            {
                string type = field.FieldType == typeof(float4) ? "float4" : field.FieldType == typeof(uint4) ? "uint4" : field.FieldType.Name;
                csharpMembers.Add(type + " " + field.Name);
            }

            CollectionAssert.AreEqual(csharpMembers, hlslMembers, hlslName + ": HLSL members differ from " + csharpType.Name);
        }

        // ---- reading the includes ---------------------------------------------------------------------

        static string PackagePath()
        {
            PackageInfo info = PackageInfo.FindForAssembly(typeof(AvbdShaderSet).Assembly);
            Assert.IsNotNull(info, "The runtime assembly is not part of a package.");
            return info.resolvedPath;
        }

        static string ReadInclude(string fileName)
        {
            string path = Path.Combine(PackagePath(), "Shaders", "Include", fileName);
            Assert.IsTrue(File.Exists(path), path + " not found");
            return File.ReadAllText(path);
        }

        static string StripComment(string line)
        {
            int comment = line.IndexOf("//", StringComparison.Ordinal);
            return comment >= 0 ? line.Substring(0, comment) : line;
        }

        static Dictionary<string, uint> ReadDefines(params string[] files)
        {
            var result = new Dictionary<string, uint>();
            foreach (string file in files)
            {
                foreach (string rawLine in ReadInclude(file).Split('\n'))
                {
                    Match m = Regex.Match(StripComment(rawLine).Trim(), @"^#define\s+(\w+)\s+(.+)$");
                    if (m.Success && TryEvaluate(m.Groups[2].Value.Trim(), out uint value))
                        result[m.Groups[1].Value] = value;
                }
            }
            return result;
        }

        // Handles the three forms the includes use: "(1u << 3)", "0x00FFFFFFu" and "64" / "0u".
        static bool TryEvaluate(string expression, out uint value)
        {
            Match shift = Regex.Match(expression, @"^\(?\s*(\d+)[uU]?\s*<<\s*(\d+)\s*\)?$");
            if (shift.Success)
            {
                value = uint.Parse(shift.Groups[1].Value) << int.Parse(shift.Groups[2].Value);
                return true;
            }

            Match hex = Regex.Match(expression, @"^0[xX]([0-9A-Fa-f]+)[uU]?$");
            if (hex.Success)
            {
                value = Convert.ToUInt32(hex.Groups[1].Value, 16);
                return true;
            }

            Match dec = Regex.Match(expression, @"^(\d+)[uU]?$");
            if (dec.Success)
            {
                value = uint.Parse(dec.Groups[1].Value);
                return true;
            }

            value = 0;
            return false;
        }
    }
}
