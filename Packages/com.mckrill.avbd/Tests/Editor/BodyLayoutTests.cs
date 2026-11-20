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

        // ---- constraint records -------------------------------------------------------------------------

        [Test]
        public void ConstraintRecord_HasTheDocumentedSize_AndFollowsRuleP3()
        {
            Assert.AreEqual(ConstraintRecordStrides.Constraint, UnsafeUtility.SizeOf<ConstraintRecord>());
            Assert.AreEqual(160, ConstraintRecordStrides.Constraint);
            Assert.AreEqual(0, ConstraintRecordStrides.Constraint % 32, "constraints are gathered by id, so the stride must be a multiple of 32 (rule P3)");
            Assert.Less(ConstraintRecordStrides.Constraint, 2048);
        }

        [Test]
        public void ConstraintFlags_AreDistinctSingleBitsInTheLow24_AndPackWithTheKind()
        {
            ConstraintFlags[] flags =
            {
                ConstraintFlags.HardLinear, ConstraintFlags.HardAngular, ConstraintFlags.Fracturable, ConstraintFlags.Broken, ConstraintFlags.Stick,
                ConstraintFlags.Plastic, ConstraintFlags.BendBreak,
            };
            uint seen = 0;
            foreach (ConstraintFlags flag in flags)
            {
                uint bit = (uint)flag;
                Assert.AreEqual(1, math.countbits(bit), flag + " is not a single bit");
                Assert.AreEqual(0u, bit & ~ConstraintFlagLayout.FlagsMask, flag + " is outside the low 24 bits");
                Assert.AreEqual(0u, seen & bit, flag + " overlaps another flag");
                seen |= bit;
            }

            uint packed = ConstraintFlagLayout.Pack(ConstraintFlags.HardLinear | ConstraintFlags.Broken, ConstraintKind.Spring);
            Assert.AreEqual(ConstraintFlags.HardLinear | ConstraintFlags.Broken, ConstraintFlagLayout.FlagsOf(packed));
            Assert.AreEqual(ConstraintKind.Spring, ConstraintFlagLayout.KindOf(packed));
        }

        [Test]
        public void Hlsl_ConstraintDefinitionsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdConstraintRecords.hlsl");

            Assert.AreEqual((uint)ConstraintFlags.HardLinear, d["AVBD_CONSTRAINT_FLAG_HARD_LIN"]);
            Assert.AreEqual((uint)ConstraintFlags.HardAngular, d["AVBD_CONSTRAINT_FLAG_HARD_ANG"]);
            Assert.AreEqual((uint)ConstraintFlags.Fracturable, d["AVBD_CONSTRAINT_FLAG_FRACTURABLE"]);
            Assert.AreEqual((uint)ConstraintFlags.Broken, d["AVBD_CONSTRAINT_FLAG_BROKEN"]);
            Assert.AreEqual((uint)ConstraintFlags.Plastic, d["AVBD_CONSTRAINT_FLAG_PLASTIC"]);
            Assert.AreEqual((uint)ConstraintFlags.BendBreak, d["AVBD_CONSTRAINT_FLAG_BEND_BREAK"]);
            Assert.AreEqual(ConstraintFlagLayout.FlagsMask, d["AVBD_CONSTRAINT_FLAGS_MASK"]);
            Assert.AreEqual((uint)ConstraintFlagLayout.KindShift, d["AVBD_CONSTRAINT_KIND_SHIFT"]);
            Assert.AreEqual((uint)ConstraintKind.Joint, d["AVBD_CONSTRAINT_KIND_JOINT"]);
            Assert.AreEqual((uint)ConstraintKind.Spring, d["AVBD_CONSTRAINT_KIND_SPRING"]);

            AssertStructMatches(ReadInclude("AvbdConstraintRecords.hlsl"), "ConstraintRecord", typeof(ConstraintRecord));
        }

        [Test]
        public void ExplosionRecord_HasTheDocumentedSize_AndMatchesTheHlslStruct()
        {
            Assert.AreEqual(32, UnsafeUtility.SizeOf<ExplosionRecord>());
            Assert.AreEqual((uint)AvbdSolver.MaxExplosionsPerStep, ReadDefines("AvbdExplosions.hlsl")["AVBD_MAX_EXPLOSIONS"]);
            AssertStructMatches(ReadInclude("AvbdExplosions.hlsl"), "ExplosionRecord", typeof(ExplosionRecord), allowScalars: true);
        }

        [Test]
        public void ForceFieldRecord_HasTheDocumentedSize_AndMatchesTheHlslStruct()
        {
            Assert.AreEqual(64, UnsafeUtility.SizeOf<ForceFieldRecord>());
            Dictionary<string, uint> d = ReadDefines("AvbdForceFields.hlsl");
            Assert.AreEqual((uint)AvbdSolver.MaxForceFields, d["AVBD_MAX_FORCE_FIELDS"]);
            Assert.AreEqual(0u, d["AVBD_FORCE_NONE"]);
            Assert.AreEqual((uint)ForceKind.Constant, d["AVBD_FORCE_CONSTANT"]);
            Assert.AreEqual((uint)ForceKind.GravityScale, d["AVBD_FORCE_GRAVITY_SCALE"]);
            Assert.AreEqual((uint)ForceKind.Drag, d["AVBD_FORCE_DRAG"]);
            Assert.AreEqual((uint)ForceKind.Wing, d["AVBD_FORCE_WING"]);
            Assert.AreEqual((uint)ForceKind.Spring, d["AVBD_FORCE_SPRING"]);
            Assert.AreEqual(1024u, d["AVBD_FORCE_FIXED_POINT"]);
            Assert.AreEqual((uint)ForceSpace.Body, d["AVBD_FORCE_FLAG_BODY_SPACE"]);
            AssertStructMatches(ReadInclude("AvbdForceFields.hlsl"), "ForceFieldRecord", typeof(ForceFieldRecord));
        }

        [Test]
        public void PlasticRecord_HasTheDocumentedSize_AndMatchesTheHlslStruct()
        {
            Assert.AreEqual(PlasticRecordStrides.Plastic, UnsafeUtility.SizeOf<PlasticRecord>());
            Assert.AreEqual(64, PlasticRecordStrides.Plastic);
            Assert.AreEqual(0, PlasticRecordStrides.Plastic % 32, "plastic records are gathered by slot, so the stride must be a multiple of 32 (rule P3)");

            AssertStructMatches(ReadInclude("AvbdPlasticRecords.hlsl"), "PlasticRecord", typeof(PlasticRecord));
        }

        [Test]
        public void MakePlastic_FlagsTheJointAndStartsAtTheIdentityRest()
        {
            ConstraintRecord joint = ConstraintRecordFactory.Joint(1u, 2u, float3.zero, float3.zero, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, 1f);

            PlasticRecord p = ConstraintRecordFactory.MakePlastic(ref joint, 7u, 50f, float.PositiveInfinity);
            ConstraintFlags flags = ConstraintFlagLayout.FlagsOf(joint.ident.z);
            Assert.IsTrue((flags & ConstraintFlags.Plastic) != 0);
            Assert.IsTrue((flags & ConstraintFlags.BendBreak) == 0, "no bend limit, no bend-break flag");
            Assert.AreEqual(ConstraintKind.Joint, ConstraintFlagLayout.KindOf(joint.ident.z));
            Assert.AreEqual(7u, joint.ident.w);
            Assert.AreEqual(new float4(0f, 0f, 0f, 1f), p.rest);
            Assert.AreEqual(new float4(0f, 0f, 0f, 1f), p.restStart);
            Assert.AreEqual(50f, p.limits.x);

            ConstraintRecord limited = ConstraintRecordFactory.Joint(1u, 2u, float3.zero, float3.zero, float.PositiveInfinity, float.PositiveInfinity, float.PositiveInfinity, 1f);
            PlasticRecord q = ConstraintRecordFactory.MakePlastic(ref limited, 0u, 50f, 0.5f);
            Assert.IsTrue((ConstraintFlagLayout.FlagsOf(limited.ident.z) & ConstraintFlags.BendBreak) != 0);
            Assert.AreEqual(0.5f, q.limits.y);
        }

        [Test]
        public void ContactRecords_HaveTheStridesOfRuleP3_AndMatchTheHlslStructs()
        {
            Assert.AreEqual(ContactRecordStrides.Pair, UnsafeUtility.SizeOf<ContactPairRecord>());
            Assert.AreEqual(ContactRecordStrides.Point, UnsafeUtility.SizeOf<ContactPointRecord>());
            Assert.AreEqual(0, ContactRecordStrides.Pair % 32, "a pair record is a multiple of 32 B");
            Assert.AreEqual(0, ContactRecordStrides.Point % 32, "a point record is a multiple of 32 B");
            Assert.LessOrEqual(ContactLayout.MaxContactsPerPair, 32, "the stick bits fit in the w word of the pair");

            string text = ReadInclude("AvbdContactRecords.hlsl");
            AssertStructMatches(text, "ContactPairRecord", typeof(ContactPairRecord));
            AssertStructMatches(text, "ContactPointRecord", typeof(ContactPointRecord));
            Assert.AreEqual(0xFFu, ReadDefines("AvbdContactRecords.hlsl")["AVBD_CONTACT_STICK_MASK"]);
        }

        [Test]
        public void ContactPoint_RoundTripsThroughTheOracleRecord()
        {
            var pair = new ContactPairRecord
            {
                ident = new uint4(7u, 3u, 2u, 0b10u),
                normalFriction = new float4(0f, 0f, 1f, 0.5f),
            };
            ContactPointRecord point = ContactCodec.Make(new float3(1f, 2f, 3f), new float3(4f, 5f, 6f), 123456, new float3(7f, 8f, 9f),
                new float3(10f, 11f, 12f), new float3(13f, 14f, 15f));

            Assert.AreEqual(123456, ContactCodec.Key(point));
            Assert.AreEqual(-1, ContactCodec.Key(ContactCodec.Make(float3.zero, float3.zero, -1, float3.zero, float3.zero, float3.zero)), "a key with the sign bit set (a NaN as a float) keeps its bits");
            Assert.AreEqual(new float3(7f, 8f, 9f), ContactCodec.C0(point));
            Assert.AreEqual(new float3(10f, 11f, 12f), ContactCodec.Penalty(point));
            Assert.AreEqual(new float3(13f, 14f, 15f), ContactCodec.Lambda(point));

            ConstraintRecord r1 = ContactCodec.ToRecord(pair, point, 1);
            Assert.AreEqual(ConstraintFlags.Stick, ConstraintFlagLayout.FlagsOf(r1.ident.z), "bit 1 of the pair's stick bits");
            Assert.AreEqual(ConstraintKind.Contact, ConstraintFlagLayout.KindOf(r1.ident.z));
            Assert.AreEqual(new uint4(7u, 3u, r1.ident.z, 123456u), r1.ident);
            Assert.AreEqual(new float4(1f, 2f, 3f, 0.5f), r1.anchorA);
            Assert.AreEqual(new float4(0f, 0f, 1f, 0f), r1.c0Ang);
            Assert.AreEqual(ConstraintFlags.None, ConstraintFlagLayout.FlagsOf(ContactCodec.ToRecord(pair, point, 0).ident.z), "bit 0 is clear");

            ContactPointRecord back = ContactCodec.FromRecord(r1);
            Assert.AreEqual(ContactCodec.RA(point), ContactCodec.RA(back));
            Assert.AreEqual(ContactCodec.Key(point), ContactCodec.Key(back));
            Assert.AreEqual(point.anchorB, back.anchorB);
            Assert.AreEqual(point.c0Penalty, back.c0Penalty);
            Assert.AreEqual(point.penaltyLambda, back.penaltyLambda);
        }

        [Test]
        public void Hlsl_ContactConstantsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdContacts.hlsl");
            Assert.AreEqual((uint)ContactLayout.MaxContactsPerPair, d["AVBD_MAX_CONTACTS"]);

            // The margin and the threshold are floats; compare them as text-parsed values.
            string text = ReadInclude("AvbdContacts.hlsl");
            Assert.AreEqual(ContactLayout.CollisionMargin, ReadFloatDefine(text, "AVBD_COLLISION_MARGIN"));
            Assert.AreEqual(ContactLayout.StickThreshold, ReadFloatDefine(text, "AVBD_STICK_THRESHOLD"));
        }

        [Test]
        public void Hlsl_ManifoldPoolConstantsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdContacts.hlsl");

            Assert.AreEqual((uint)ContactLayout.PoolFreeTopWord, d["AVBD_NP_FREE_TOP"]);
            Assert.AreEqual((uint)ContactLayout.PoolOverflowWord, d["AVBD_NP_OVERFLOW"]);
            Assert.AreEqual((uint)ContactLayout.PoolHighWaterWord, d["AVBD_NP_HIGH_WATER"]);
            Assert.AreEqual((uint)ContactLayout.PoolControlWords, d["AVBD_NP_CONTROL_WORDS"]);
            Assert.Less(Math.Max(ContactLayout.PoolFreeTopWord, Math.Max(ContactLayout.PoolOverflowWord, ContactLayout.PoolHighWaterWord)),
                ContactLayout.PoolControlWords, "every control word is inside the buffer");

            Assert.AreEqual((uint)ContactLayout.PoolArgsRetire, d["AVBD_NP_ARGS_RETIRE"]);
            Assert.AreEqual((uint)ContactLayout.PoolArgsAllocateDynamic, d["AVBD_NP_ARGS_ALLOCATE_DYNAMIC"]);
            Assert.AreEqual((uint)ContactLayout.PoolArgsAllocateStatic, d["AVBD_NP_ARGS_ALLOCATE_STATIC"]);
            Assert.AreEqual((uint)ContactLayout.PoolArgsNarrowphase, d["AVBD_NP_ARGS_NARROWPHASE"]);
            Assert.AreEqual((uint)ContactLayout.PoolArgsDual, d["AVBD_NP_ARGS_DUAL"]);
            Assert.AreEqual((uint)ContactLayout.PoolArgsCount, d["AVBD_NP_ARGS_COUNT"]);
            Assert.AreEqual(ContactLayout.PoolArgsCount - 1, ContactLayout.PoolArgsDual, "the entries are numbered without a gap");
        }

        [Test]
        public void Hlsl_IndirectConstantsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdIndirect.hlsl", "AvbdColoringParams.hlsl");

            Assert.AreEqual((uint)AvbdIndirect.ArgsStride, d["AVBD_INDIRECT_ARGS_STRIDE"]);
            Assert.AreEqual((uint)AvbdIndirect.RowGroups, d["AVBD_INDIRECT_ROW_GROUPS"]);
            Assert.AreEqual(3 * sizeof(uint), AvbdIndirect.ArgsStride, "groups x, y and z");
            Assert.AreEqual(65535, AvbdIndirect.RowGroups, "the limit of one dimension of a dispatch");

            // The coloring has its own copy of the same arithmetic in ColorPlan.
            Assert.AreEqual(d["AVBD_INDIRECT_ARGS_STRIDE"], d["AVBD_COLOR_ARGS_STRIDE"]);
            Assert.AreEqual(d["AVBD_INDIRECT_ROW_GROUPS"], d["AVBD_COLOR_ROW_GROUPS"]);
        }

        [Test]
        public void ContactPairIndex_RoundTripsAndIsDense()
        {
            long expected = 0;
            for (int hi = 1; hi < 600; hi++)
                for (int lo = 0; lo < hi; lo++, expected++)
                {
                    Assert.AreEqual(expected, ContactLayout.PairIndex(hi, lo));
                    Assert.AreEqual(expected, ContactLayout.PairIndex(lo, hi));
                    ContactLayout.BodiesOfPair(expected, out int a, out int b);
                    Assert.AreEqual(hi, a);
                    Assert.AreEqual(lo, b);
                }
            Assert.AreEqual(expected, ContactLayout.PairCount(600));
        }

        static float ReadFloatDefine(string text, string name)
        {
            var m = System.Text.RegularExpressions.Regex.Match(text, @"#define\s+" + name + @"\s+([0-9.eE+-]+)");
            Assert.IsTrue(m.Success, "define " + name + " not found");
            return float.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
        }

        [Test]
        public void Hlsl_ColoringConstantsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdColoringParams.hlsl");

            Assert.AreEqual((uint)AvbdGraphColoring.MaxColors, d["AVBD_COLOR_MAX"]);
            Assert.AreEqual(AvbdGraphColoring.Pending, d["AVBD_COLOR_PENDING"]);
            Assert.AreEqual((uint)AvbdGraphColoring.ControlWords, d["AVBD_COLOR_CONTROL_WORDS"]);
            Assert.AreEqual((uint)AvbdGraphColoring.CountsOffset, d["AVBD_COLOR_COUNTS_OFFSET"]);
            Assert.AreEqual((uint)AvbdGraphColoring.ColorCountWord, d["AVBD_COLOR_COUNT_WORD"]);
            Assert.AreEqual((uint)AvbdGraphColoring.CursorsOffset, d["AVBD_COLOR_CURSORS_OFFSET"]);
            Assert.AreEqual((uint)AvbdGraphColoring.StartsOffset, d["AVBD_COLOR_STARTS_OFFSET"]);
            Assert.AreEqual((uint)AvbdGraphColoring.WorkWords, d["AVBD_COLOR_WORK_WORDS"]);
            Assert.AreEqual((uint)AvbdGraphColoring.StatReleased, d["AVBD_COLOR_STAT_RELEASED"]);
            Assert.AreEqual((uint)AvbdGraphColoring.StatLastRound, d["AVBD_COLOR_STAT_LAST_ROUND"]);
            Assert.AreEqual((uint)AvbdGraphColoring.StatFinished, d["AVBD_COLOR_STAT_FINISHED"]);
            Assert.AreEqual((uint)AvbdGraphColoring.ArgsStride, d["AVBD_COLOR_ARGS_STRIDE"]);
            Assert.AreEqual((uint)AvbdGraphColoring.RowGroups, d["AVBD_COLOR_ROW_GROUPS"]);
            Assert.AreEqual((uint)AvbdGraphColoring.PrimalLanes, d["AVBD_PRIMAL_LANES"]);
            Assert.AreEqual((uint)AvbdGraphColoring.RoundLanes, d["AVBD_COLOR_LANES"]);
            Assert.AreEqual(0, AvbdConst.GroupSize % AvbdGraphColoring.RoundLanes, "the lanes of a body are in one thread group");
            Assert.AreEqual(0, AvbdConst.GroupSize % AvbdGraphColoring.PrimalLanes, "the lanes of a body are in one thread group");
            Assert.AreEqual(0, AvbdGraphColoring.PrimalLanes & (AvbdGraphColoring.PrimalLanes - 1), "a power of two");
            Assert.AreEqual((uint)GraphColoringOrder.SlotOrder, d["AVBD_COLOR_ORDER_SLOT"]);
            Assert.AreEqual((uint)GraphColoringOrder.Hashed, d["AVBD_COLOR_ORDER_HASHED"]);

            // The four sections of the work buffer follow each other without a gap, and a color is never confused with a marker.
            Assert.AreEqual(AvbdGraphColoring.ControlWords, AvbdGraphColoring.CountsOffset);
            Assert.AreEqual(AvbdGraphColoring.CountsOffset + AvbdGraphColoring.MaxColors, AvbdGraphColoring.CursorsOffset);
            Assert.AreEqual(AvbdGraphColoring.CursorsOffset + AvbdGraphColoring.MaxColors, AvbdGraphColoring.StartsOffset);
            Assert.AreEqual(AvbdGraphColoring.StartsOffset + AvbdGraphColoring.MaxColors, AvbdGraphColoring.StatReleased);
            Assert.AreEqual(AvbdGraphColoring.StatReleased + 1, AvbdGraphColoring.StatLastRound);
            Assert.AreEqual(AvbdGraphColoring.StatLastRound + 1, AvbdGraphColoring.StatFinished);
            Assert.LessOrEqual(AvbdGraphColoring.StatFinished + 1, AvbdGraphColoring.WorkWords);
            Assert.Less(AvbdGraphColoring.ColorCountWord, AvbdGraphColoring.ControlWords);
            Assert.AreEqual(3 * sizeof(uint), AvbdGraphColoring.ArgsStride, "groups x, y and z");
            Assert.AreEqual(65535, AvbdGraphColoring.RowGroups, "the limit of one dimension of a dispatch");
            Assert.Greater(AvbdGraphColoring.Pending, (uint)AvbdGraphColoring.MaxColors);
            Assert.AreNotEqual(AvbdConst.Invalid, AvbdGraphColoring.Pending);
        }

        [Test]
        public void Hlsl_ListConstantsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdListParams.hlsl");

            Assert.AreEqual((uint)AvbdConstraintLists.ControlTotal, d["AVBD_LIST_CONTROL_TOTAL"]);
            Assert.AreEqual((uint)AvbdConstraintLists.ControlBroken, d["AVBD_LIST_CONTROL_BROKEN"]);
            Assert.AreEqual((uint)AvbdConstraintLists.ControlInvalid, d["AVBD_LIST_CONTROL_INVALID"]);
            Assert.AreEqual((uint)AvbdConstraintLists.ControlWords, d["AVBD_LIST_CONTROL_WORDS"]);
            Assert.AreEqual((uint)AvbdConstraintLists.ScanGroup, d["AVBD_LIST_SCAN_GROUP"]);
            Assert.AreEqual((uint)AvbdConstraintLists.SortLanes, d["AVBD_LIST_SORT_LANES"]);
            Assert.AreEqual(0, AvbdConst.GroupSize % AvbdConstraintLists.SortLanes, "the lanes of a body are in one thread group");
            Assert.AreEqual((uint)ContactLayout.MaxContactsPerPair, d["AVBD_LIST_CONTACTS_PER_SLOT"]);
            Assert.AreEqual(AvbdConstraintLists.ContactFlag, d["AVBD_LIST_CONTACT_FLAG"]);
            Assert.Less((uint)AvbdConstraintBuffers.MaxConstraints, AvbdConstraintLists.ContactFlag, "constraint ids stay below the contact flag");
            Assert.Less((uint)AvbdConstraintBuffers.MaxContactManifolds, AvbdConstraintLists.ContactFlag, "manifold slots stay below the contact flag");
        }

        [Test]
        public void Hlsl_BroadphaseConstantsMatchTheCSharpMirrors()
        {
            Dictionary<string, uint> d = ReadDefines("AvbdBroadphaseParams.hlsl");

            Assert.AreEqual((uint)PairKind.DynamicDynamic, d["AVBD_PAIR_DYNAMIC_DYNAMIC"]);
            Assert.AreEqual((uint)PairKind.DynamicStatic, d["AVBD_PAIR_DYNAMIC_STATIC"]);
            Assert.AreEqual((uint)AvbdBroadphase.MaxRadiusWord, d["AVBD_BP_MAX_RADIUS"]);
            Assert.AreEqual((uint)AvbdBroadphase.LargeCountWord, d["AVBD_BP_LARGE_COUNT"]);
            Assert.AreEqual((uint)AvbdBroadphase.DynamicPairsWord, d["AVBD_BP_DYNAMIC_PAIRS"]);
            Assert.AreEqual((uint)AvbdBroadphase.StaticPairsWord, d["AVBD_BP_STATIC_PAIRS"]);
            Assert.AreEqual((uint)AvbdBroadphase.SmallBinWord, d["AVBD_BP_SMALL_BIN"]);
            Assert.AreEqual((uint)AvbdBroadphase.FineBinWord, d["AVBD_BP_FINE_BIN"]);
            Assert.AreEqual((uint)AvbdBroadphase.TwoLevelsWord, d["AVBD_BP_TWO_LEVELS"]);
            Assert.AreEqual((uint)AvbdBroadphase.CoarseCellWord, d["AVBD_BP_COARSE_CELL"]);
            Assert.AreEqual((uint)AvbdBroadphase.FineCellWord, d["AVBD_BP_FINE_CELL"]);
            Assert.AreEqual((uint)AvbdBroadphase.LiveCountWord, d["AVBD_BP_LIVE_COUNT"]);
            Assert.AreEqual((uint)AvbdBroadphase.SolvedCountWord, d["AVBD_BP_SOLVED_COUNT"]);
            Assert.AreEqual((uint)AvbdBroadphase.StatWords, d["AVBD_BP_STAT_WORDS"]);
            Assert.AreEqual((uint)AvbdBroadphase.HistAllWord, d["AVBD_BP_HIST_ALL"]);
            Assert.AreEqual((uint)AvbdBroadphase.HistSolvedWord, d["AVBD_BP_HIST_SOLVED"]);
            Assert.AreEqual((uint)AvbdBroadphase.ControlWords, d["AVBD_BP_CONTROL_WORDS"]);
            Assert.AreEqual((uint)AvbdBroadphase.BinsPerOctave, d["AVBD_BP_BINS_PER_OCTAVE"]);
            Assert.AreEqual((uint)AvbdBroadphase.BinOctaves, d["AVBD_BP_BIN_OCTAVES"]);
            Assert.AreEqual((uint)AvbdBroadphase.RadiusBins, d["AVBD_BP_RADIUS_BINS"]);
            Assert.AreEqual((uint)AvbdBroadphase.LargeFactorBins, d["AVBD_BP_LARGE_FACTOR_BINS"]);
            Assert.AreEqual((uint)AvbdBroadphase.MaxLarge, d["AVBD_BP_MAX_LARGE"]);
            Assert.AreEqual((uint)AvbdBroadphase.ScanGroup, d["AVBD_BP_SCAN_GROUP"]);
            Assert.AreEqual(AvbdBroadphase.StaticCellFlag, d["AVBD_BP_STATIC_FLAG"]);
            Assert.AreEqual(AvbdBroadphase.LargeCellFlag, d["AVBD_BP_LARGE_FLAG"]);
            Assert.AreEqual(AvbdBroadphase.CoarseCellFlag, d["AVBD_BP_COARSE_FLAG"]);
            StringAssert.Contains("AVBD_BP_CELL_MARGIN " + AvbdBroadphase.CellMargin.ToString("0.0#", System.Globalization.CultureInfo.InvariantCulture),
                ReadInclude("AvbdBroadphaseParams.hlsl"));

            StringAssert.Contains("AVBD_BP_SEPARATION_PAD " + AvbdBroadphase.SeparationPad.ToString("0.0##", System.Globalization.CultureInfo.InvariantCulture),
                ReadInclude("AvbdBroadphaseParams.hlsl"));

            Assert.AreEqual(AvbdBroadphase.PairStride, UnsafeUtility.SizeOf<PairRecord>());
            AssertStructMatches(ReadInclude("AvbdBroadphaseParams.hlsl"), "PairRecord", typeof(PairRecord));
        }

        // allowScalars is for the exception to rule P1 (see the layout rules): a record with float, uint and float2 members that fill whole 16 B rows. Every other record
        // must be made of float4 and uint4 only.
        static void AssertStructMatches(string hlslSource, string hlslName, Type csharpType, bool allowScalars = false)
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
                bool vectorWord = member.Groups[1].Value == "float4" || member.Groups[1].Value == "uint4";
                bool scalar = member.Groups[1].Value == "float" || member.Groups[1].Value == "uint" || member.Groups[1].Value == "float2";
                Assert.IsTrue(vectorWord || (allowScalars && scalar),
                    hlslName + "." + member.Groups[2].Value + " is " + member.Groups[1].Value + " (rule P1 allows only float4 and uint4)");
                hlslMembers.Add(member.Groups[1].Value + " " + member.Groups[2].Value);
            }

            var csharpMembers = new List<string>();
            foreach (FieldInfo field in csharpType.GetFields(BindingFlags.Public | BindingFlags.Instance))
            {
                string type = field.FieldType == typeof(float4) ? "float4" : field.FieldType == typeof(uint4) ? "uint4"
                    : allowScalars && field.FieldType == typeof(float) ? "float" : allowScalars && field.FieldType == typeof(uint) ? "uint" : field.FieldType.Name;
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
