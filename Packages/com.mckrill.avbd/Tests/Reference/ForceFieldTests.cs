using System;
using System.Collections.Generic;
using NUnit.Framework;
using Unity.Mathematics;
using UnityEngine;
using McKrill.Avbd.Reference;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// Force fields: forces that act on the bodies of a group (the group mask of the body is its label), applied in stage P. The CPU oracle is checked
    /// against numbers worked out by hand, and the GPU kernel against the oracle.
    /// </summary>
    public class ForceFieldTests
    {
        const int Capacity = 128;
        const uint Wings = 1u;
        const uint Body = 2u;

        // ---- the oracle, by hand --------------------------------------------------------------------------

        [Test]
        public void Force_ActsOnlyOnTheBodiesOfItsGroup_NotOnOtherGroupsOrStaticOrKinematicBodies()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            BodyDescriptor[] scene =
            {
                Box(new float3(0f), 1f, Wings),                              // 0: in the group
                Box(new float3(3f, 0f, 0f), 1f, Body),                       // 1: not in the group
                Box(new float3(6f, 0f, 0f), 1f, Wings | Body),               // 2: in two groups, one of them the field's
                Kinematic(new float3(9f, 0f, 0f), Wings),                    // 3: kinematic
                Static(new float3(12f, 0f, 0f), Wings),                      // 4: static
            };
            ForceFieldRecord[] table = Table(ForceFieldDescriptor.Force(Wings, new float3(0f, 0f, 60f)));

            CpuBodyArrays plain = Scene(scene);
            CpuBodyArrays forced = Scene(scene);
            CpuPredict.Run(plain, settings, 5);
            CpuPredict.Run(forced, settings, 5, null, 0, table, 1);

            float push = 60f / 1f * settings.dt * settings.dt; // F / m dt^2, with 1 kg
            foreach (int slot in new[] { 0, 2 })
            {
                Assert.AreEqual(plain.Inertial[slot].position.z + push, forced.Inertial[slot].position.z, 1e-5f, "slot " + slot);
                Assert.AreEqual(plain.Inertial[slot].position.x, forced.Inertial[slot].position.x, 0f, "slot " + slot + " x");
            }
            foreach (int slot in new[] { 1, 3, 4 })
            {
                Assert.AreEqual(plain.Inertial[slot].position, forced.Inertial[slot].position, "slot " + slot + " is not moved");
                Assert.AreEqual(plain.State[slot].position, forced.State[slot].position, "slot " + slot + " pose");
            }
        }

        [Test]
        public void Force_IsPerBody_AndDividedByTheMass()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            BodyDescriptor light = Box(new float3(0f), 1f, Wings);      // 1 kg
            BodyDescriptor heavy = Box(new float3(3f, 0f, 0f), 4f, Wings); // 4 kg
            CpuBodyArrays bodies = Scene(light, heavy);
            CpuBodyArrays plain = Scene(light, heavy);
            CpuPredict.Run(plain, settings, 2);
            CpuPredict.Run(bodies, settings, 2, null, 0, Table(ForceFieldDescriptor.Force(Wings, new float3(0f, 8f, 0f))), 1);

            float dt2 = settings.dt * settings.dt;
            Assert.AreEqual(8f * dt2, bodies.Inertial[0].position.y - plain.Inertial[0].position.y, 1e-6f);
            Assert.AreEqual(2f * dt2, bodies.Inertial[1].position.y - plain.Inertial[1].position.y, 1e-6f, "each body gets the whole force: 8 N on 4 kg");
        }

        [Test]
        public void Force_InTheBodyFrame_TurnsWithTheBody()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            BodyDescriptor turned = Box(new float3(0f), 1f, Wings);
            turned.rotation = quaternion.RotateY(math.radians(90f)); // its z axis points along the world's x
            CpuBodyArrays bodies = Scene(turned);
            CpuBodyArrays plain = Scene(turned);
            CpuPredict.Run(plain, settings, 1);
            CpuPredict.Run(bodies, settings, 1, null, 0, Table(ForceFieldDescriptor.Force(Wings, new float3(0f, 0f, 10f), ForceSpace.Body)), 1);

            float3 delta = (bodies.Inertial[0].position - plain.Inertial[0].position).xyz;
            float dt2 = settings.dt * settings.dt;
            Assert.AreEqual(10f * dt2, delta.x, 1e-6f);
            Assert.AreEqual(0f, delta.y, 1e-6f);
            Assert.AreEqual(0f, delta.z, 1e-6f);
        }

        [Test]
        public void GravityScale_MakesGravityWeakerForTheGroup_AndLeavesTheRestAlone()
        {
            AvbdSolverSettings settings = Settings(new float3(0f, -10f, 0f));
            var scene = new[]
            {
                Box(new float3(0f), 1f, Wings),            // light
                Box(new float3(3f, 0f, 0f), 1f, Body),     // normal
                Box(new float3(6f, 0f, 0f), 1f, Wings),    // light too
            };
            CpuBodyArrays bodies = Scene(scene);
            CpuPredict.Run(bodies, settings, 3, null, 0, Table(ForceFieldDescriptor.GravityScale(Wings, 0.25f)), 1);

            float dt2 = settings.dt * settings.dt;
            Assert.AreEqual(-10f * 0.25f * dt2, bodies.Inertial[0].position.y, 1e-6f, "a quarter of gravity");
            Assert.AreEqual(-10f * dt2, bodies.Inertial[1].position.y, 1e-6f, "normal gravity");

            // Scale 0: weightless. Scale -1: falls up. They do not need a gravity of their own.
            CpuBodyArrays weightless = Scene(scene);
            CpuPredict.Run(weightless, settings, 3, null, 0, Table(ForceFieldDescriptor.GravityScale(Wings, 0f)), 1);
            Assert.AreEqual(0f, weightless.Inertial[0].position.y, 1e-6f);
            CpuBodyArrays up = Scene(scene);
            CpuPredict.Run(up, settings, 3, null, 0, Table(ForceFieldDescriptor.GravityScale(Wings, -1f)), 1);
            Assert.AreEqual(10f * dt2, up.Inertial[0].position.y, 1e-6f);
        }

        [Test]
        public void Fields_Add_WhenABodyIsInSeveralGroups()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            BodyDescriptor both = Box(new float3(0f), 1f, Wings | Body);
            CpuBodyArrays bodies = Scene(both);
            CpuPredict.Run(bodies, settings, 1, null, 0, Table(
                ForceFieldDescriptor.Force(Wings, new float3(0f, 6f, 0f)),
                ForceFieldDescriptor.Force(Body, new float3(0f, 0f, 3f))), 2);

            float dt2 = settings.dt * settings.dt;
            Assert.AreEqual(6f * dt2, bodies.Inertial[0].position.y, 1e-6f);
            Assert.AreEqual(3f * dt2, bodies.Inertial[0].position.z, 1e-6f);
        }

        [Test]
        public void Drag_PushesAgainstTheVelocityRelativeToTheWind()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            BodyDescriptor moving = Box(new float3(0f), 2f, Wings); // 2 kg
            moving.linearVelocity = new float3(10f, 0f, 0f);
            CpuBodyArrays plain = Scene(moving);
            CpuPredict.Run(plain, settings, 1);

            float dt2 = settings.dt * settings.dt;

            // Still air: force = -(1 + 0.5 * 10) * 10 = -60 N, so -30 m/s^2 on 2 kg.
            CpuBodyArrays still = Scene(moving);
            CpuPredict.Run(still, settings, 1, null, 0, Table(ForceFieldDescriptor.Drag(Wings, 1f, 0.5f)), 1);
            Assert.AreEqual(-30f * dt2, still.Inertial[0].position.x - plain.Inertial[0].position.x, 1e-5f);

            // Wind of 4 m/s along x: the body moves through the air at 6 m/s. Force = -(1 + 0.5 * 6) * 6 = -24 N, so -12 m/s^2.
            CpuBodyArrays windy = Scene(moving);
            CpuPredict.Run(windy, settings, 1, null, 0, Table(ForceFieldDescriptor.Drag(Wings, 1f, 0.5f, new float3(4f, 0f, 0f))), 1);
            Assert.AreEqual(-12f * dt2, windy.Inertial[0].position.x - plain.Inertial[0].position.x, 1e-5f);

            // A body at rest in the wind is pushed along it. Wind 3 m/s along z: force = -(1 + 0.5 * 3) * (-3) = 7.5 N, so 3.75 m/s^2.
            BodyDescriptor resting = Box(new float3(0f), 2f, Wings);
            CpuBodyArrays calm = Scene(resting);
            CpuBodyArrays blown = Scene(resting);
            CpuPredict.Run(calm, settings, 1);
            CpuPredict.Run(blown, settings, 1, null, 0, Table(ForceFieldDescriptor.Drag(Wings, 1f, 0.5f, new float3(0f, 0f, 3f))), 1);
            Assert.AreEqual(3.75f * dt2, blown.Inertial[0].position.z - calm.Inertial[0].position.z, 1e-5f);
        }

        [Test]
        public void ShieldedDrag_ScalesTheDragByTheJointsOfTheBody_ContactsNotCounted()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            float dt2 = settings.dt * settings.dt;
            BodyDescriptor moving = Box(new float3(0f), 2f, Wings); // 2 kg
            moving.linearVelocity = new float3(10f, 0f, 0f);
            CpuBodyArrays plain = Scene(moving);
            CpuPredict.Run(plain, settings, 1);

            // Full drag: -(1 + 0.5 * 10) * 10 = -60 N = -30 m/s^2. Fully shielded at 18 joints, fully exposed at 13 or fewer, linear in between.
            ForceFieldRecord[] table = Table(ForceFieldDescriptor.ShieldedDrag(Wings, 1f, 0.5f));
            float[] expected = { 1f, 1f, 1f, 0.8f, 0.6f, 0.4f, 0.2f, 0f, 0f }; // 0, 12, 13, 14, 15, 16, 17, 18, 20 joints
            uint[] joints = { 0u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 20u };
            for (int i = 0; i < joints.Length; i++)
            {
                CpuBodyArrays bodies = Scene(moving);
                bodies.Range[0] = new BodyRangeRecord { range = new uint4(0u, joints[i], 0u, 0u) };
                CpuPredict.Run(bodies, settings, 1, null, 0, table, 1);
                Assert.AreEqual(-30f * expected[i] * dt2, bodies.Inertial[0].position.x - plain.Inertial[0].position.x, 1e-5f, joints[i] + " joints");
            }

            // The contacts are the first entries of the list and do not count: 20 entries of which 7 are contacts is 13 joints, fully exposed.
            CpuBodyArrays touching = Scene(moving);
            touching.Range[0] = new BodyRangeRecord { range = new uint4(0u, 20u, 7u, 0u) };
            CpuPredict.Run(touching, settings, 1, null, 0, table, 1);
            Assert.AreEqual(-30f * dt2, touching.Inertial[0].position.x - plain.Inertial[0].position.x, 1e-5f);

            // A drag field without shielding ignores the joints.
            CpuBodyArrays buried = Scene(moving);
            buried.Range[0] = new BodyRangeRecord { range = new uint4(0u, 18u, 0u, 0u) };
            CpuPredict.Run(buried, settings, 1, null, 0, Table(ForceFieldDescriptor.Drag(Wings, 1f, 0.5f)), 1);
            Assert.AreEqual(-30f * dt2, buried.Inertial[0].position.x - plain.Inertial[0].position.x, 1e-5f);
        }

        [Test]
        public void ShieldedDrag_RejectsJointCountsThatCannotWork()
        {
            Assert.DoesNotThrow(() => ForceFieldDescriptor.ShieldedDrag(Wings, 1f, 0.5f).Validate());
            Assert.Throws<System.ArgumentException>(() => ForceFieldDescriptor.ShieldedDrag(Wings, 1f, 0.5f, default, 13f, 13f).Validate(), "no room between exposed and shielded");
            Assert.Throws<System.ArgumentException>(() => ForceFieldDescriptor.ShieldedDrag(Wings, 1f, 0.5f, default, 18f, -1f).Validate());
            Assert.Throws<System.ArgumentException>(() => ForceFieldDescriptor.ShieldedDrag(Wings, 1f, 0.5f, default, float.NaN, 13f).Validate());
        }

        [Test]
        public void Drag_NeverReversesTheRelativeVelocityInOneStep()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            BodyDescriptor moving = Box(new float3(0f), 1f, Wings);
            moving.linearVelocity = new float3(10f, 0f, 0f);
            CpuBodyArrays bodies = Scene(moving);
            CpuPredict.Run(bodies, settings, 1, null, 0, Table(ForceFieldDescriptor.Drag(Wings, 1e6f, 0f)), 1);

            // The drag is limited to what stops the body in one step: it ends at rest relative to the air, not flying backward.
            Assert.AreEqual(0f, bodies.Inertial[0].position.x, 1e-4f);
        }

        [Test]
        public void Wing_GivesLiftAcrossTheFlow_AndDragAlongIt_FromTheBlocksOfTheWingsOnly()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            float alpha = 0.1f;
            float speed = 20f;
            // An airplane: the wing blocks (group 1) are pitched nose up by alpha, and so are the body blocks (group 2). Only the wings get the field.
            quaternion pitch = quaternion.RotateX(-alpha); // turns the up axis (0, 1, 0) to (0, cos, -sin): the nose goes up
            BodyDescriptor wing = Box(new float3(0f), 1f, Wings);
            BodyDescriptor fuselage = Box(new float3(0f, 0f, 3f), 1f, Body);
            wing.rotation = pitch;
            fuselage.rotation = pitch;
            wing.linearVelocity = new float3(0f, 0f, speed);
            fuselage.linearVelocity = new float3(0f, 0f, speed);

            CpuBodyArrays plain = Scene(wing, fuselage);
            CpuBodyArrays flying = Scene(wing, fuselage);
            float area = 2f;
            float density = 1.2f;
            float slope = 4.5f;
            float cd0 = 0.05f;
            CpuPredict.Run(plain, settings, 2);
            CpuPredict.Run(flying, settings, 2, null, 0,
                Table(ForceFieldDescriptor.Wing(Wings, new float3(0f, 1f, 0f), area, slope, cd0, density)), 1);

            float dt2 = settings.dt * settings.dt;
            float3 wingAcceleration = (flying.Inertial[0].position - plain.Inertial[0].position).xyz / dt2; // 1 kg: the force in N
            float q = 0.5f * density * area * speed * speed;
            float sinA = math.sin(alpha);
            float cosA = math.cos(alpha);
            // force = -q (slope (v^ . n) n + cd0 v^) with v^ = (0, 0, 1) and n = (0, cos, -sin), so v^ . n = -sin.
            float3 expected = -q * (slope * -sinA * new float3(0f, cosA, -sinA) + cd0 * new float3(0f, 0f, 1f));
            AssertNear(expected, wingAcceleration, 2e-3f, "the force on the wing");
            Assert.Greater(wingAcceleration.y, 0f, "lift is up");
            Assert.Less(wingAcceleration.z, 0f, "drag is against the flight");
            Assert.AreEqual(0f, wingAcceleration.x, 1e-3f);

            Assert.AreEqual(plain.Inertial[1].position, flying.Inertial[1].position, "the fuselage makes no lift");
        }

        [Test]
        public void Wing_AtNoAngleOfAttackOnlyDrags_AndUpsideDownItPushesTheOtherWay()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            float dt2 = settings.dt * settings.dt;

            Func<float, float3> acceleration = pitchRadians =>
            {
                BodyDescriptor wing = Box(new float3(0f), 1f, Wings);
                wing.rotation = quaternion.RotateX(pitchRadians);
                wing.linearVelocity = new float3(0f, 0f, 15f);
                CpuBodyArrays plain = Scene(wing);
                CpuBodyArrays flying = Scene(wing);
                CpuPredict.Run(plain, settings, 1);
                CpuPredict.Run(flying, settings, 1, null, 0, Table(ForceFieldDescriptor.Wing(Wings, new float3(0f, 1f, 0f), 1f)), 1);
                return (flying.Inertial[0].position - plain.Inertial[0].position).xyz / dt2;
            };

            float3 level = acceleration(0f);
            Assert.AreEqual(0f, level.y, 1e-3f, "no lift at no angle");
            Assert.Less(level.z, 0f, "but some drag");
            Assert.Less(acceleration(0.2f).y, 0f, "nose down: negative lift");
            Assert.Greater(acceleration(-0.2f).y, 0f, "nose up: lift");
        }

        [Test]
        public void Wing_AtRest_GivesNothing()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            CpuBodyArrays plain = Scene(Box(new float3(0f), 1f, Wings));
            CpuBodyArrays bodies = Scene(Box(new float3(0f), 1f, Wings));
            CpuPredict.Run(plain, settings, 1);
            CpuPredict.Run(bodies, settings, 1, null, 0, Table(ForceFieldDescriptor.Wing(Wings, new float3(0f, 1f, 0f), 1f)), 1);
            Assert.AreEqual(plain.Inertial[0].position, bodies.Inertial[0].position);
        }

        [Test]
        public void EmptyTables_AndFieldsOfOtherGroups_LeaveThePredictPhaseUnchanged()
        {
            AvbdSolverSettings settings = Settings(new float3(0f, -10f, 0f));
            BodyDescriptor[] scene = { Box(new float3(0f), 1f, Body), Box(new float3(2f, 0f, 0f), 1f, Body) };
            CpuBodyArrays plain = Scene(scene);
            CpuBodyArrays empty = Scene(scene);
            CpuBodyArrays other = Scene(scene);
            CpuPredict.Run(plain, settings, 2);
            CpuPredict.Run(empty, settings, 2, null, 0, new ForceFieldRecord[AvbdSolver.MaxForceFields], 4);
            CpuPredict.Run(other, settings, 2, null, 0, Table(ForceFieldDescriptor.Force(Wings, new float3(1f, 2f, 3f))), 1);
            for (int slot = 0; slot < 2; slot++)
            {
                Assert.AreEqual(plain.Inertial[slot].position, empty.Inertial[slot].position);
                Assert.AreEqual(plain.State[slot].position, empty.State[slot].position);
                Assert.AreEqual(plain.Inertial[slot].position, other.Inertial[slot].position);
            }
        }

        [Test]
        public void Spring_PullsTheCentroidOfTheGroup_AndEveryBodyOfItGetsTheSameForce()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            BodyDescriptor[] scene =
            {
                Box(new float3(0f, 0f, 0f), 1f, Wings),
                Box(new float3(2f, 0f, 0f), 1f, Wings),
                Box(new float3(4f, 0f, 0f), 1f, Wings),     // the centroid of the three is (2, 0, 0)
                Box(new float3(100f, 0f, 0f), 1f, Body),    // another group: not in the centroid, not pulled
                Static(new float3(50f, 0f, 0f), Wings),     // static: not in the centroid, not moved
            };
            CpuBodyArrays plain = Scene(scene);
            CpuBodyArrays pulled = Scene(scene);
            CpuPredict.Run(plain, settings, 5);
            CpuPredict.Run(pulled, settings, 5, null, 0, Table(ForceFieldDescriptor.Spring(Wings, new float3(2f, 6f, 0f), 100f, 0f)), 1);

            // The centroid is 6 m below the target: every body of the group gets 100 N/m * 6 m = 600 N up, whatever its own position.
            float push = 600f * settings.dt * settings.dt;
            for (int slot = 0; slot < 3; slot++)
            {
                AssertNear(plain.Inertial[slot].position.xyz + new float3(0f, push, 0f), pulled.Inertial[slot].position.xyz, 1e-5f, "slot " + slot);
            }
            Assert.AreEqual(plain.Inertial[3].position, pulled.Inertial[3].position, "another group");
            Assert.AreEqual(plain.Inertial[4].position, pulled.Inertial[4].position, "static");
        }

        [Test]
        public void Spring_ForceIsPerBody_NeverCarriesTheGroupPastTheTarget_AndDampsTheVelocity()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            float dt = settings.dt;

            // A very stiff spring: the explicit step is limited to the distance to the target (6 m), not 1e7 * 6 / m * dt^2.
            CpuBodyArrays stiffPlain = Scene(Box(new float3(0f), 1f, Wings));
            CpuBodyArrays stiff = Scene(Box(new float3(0f), 1f, Wings));
            CpuPredict.Run(stiffPlain, settings, 1);
            CpuPredict.Run(stiff, settings, 1, null, 0, Table(ForceFieldDescriptor.Spring(Wings, new float3(0f, 6f, 0f), 1e7f, 0f)), 1);
            Assert.AreEqual(6f, stiff.Inertial[0].position.y - stiffPlain.Inertial[0].position.y, 1e-3f, "it arrives at the target and not past it");

            // A body already at the target with a velocity: only the damping acts, 2 zeta sqrt(k / m) v = 2 * 10 * v for k = 100, m = 1 and zeta = 1.
            BodyDescriptor moving = Box(new float3(0f), 1f, Wings);
            moving.linearVelocity = new float3(0f, 0f, 3f);
            CpuBodyArrays dampedPlain = Scene(moving);
            CpuBodyArrays damped = Scene(moving);
            CpuPredict.Run(dampedPlain, settings, 1);
            CpuPredict.Run(damped, settings, 1, null, 0, Table(ForceFieldDescriptor.Spring(Wings, new float3(0f), 100f, 1f)), 1);
            Assert.AreEqual(-60f * dt * dt, damped.Inertial[0].position.z - dampedPlain.Inertial[0].position.z, 1e-5f);

            // A heavy damping ratio stops the body in the step and does not turn it around.
            CpuBodyArrays stopped = Scene(moving);
            CpuPredict.Run(stopped, settings, 1, null, 0, Table(ForceFieldDescriptor.Spring(Wings, new float3(0f), 1e6f, 5f)), 1);
            Assert.AreEqual(0f, stopped.Inertial[0].position.z, 1e-5f, "v dt - v dt = 0");
        }

        [Test]
        public void Spring_WithALine_IsStifferTheFartherTheCentroidIsFromTheLine()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            float dt2 = settings.dt * settings.dt;

            // One body at the origin; the target is (3, 0, 5) and the line through it runs along z (the line of sight). The body is 3 m from the line
            // (the part of its offset across the line) and 5 m along it (which does not count).
            Func<float, float3> pull = gain =>
            {
                CpuBodyArrays plain = Scene(Box(new float3(0f), 1f, Wings));
                CpuBodyArrays pulled = Scene(Box(new float3(0f), 1f, Wings));
                CpuPredict.Run(plain, settings, 1);
                CpuPredict.Run(pulled, settings, 1, null, 0,
                    Table(ForceFieldDescriptor.Spring(Wings, new float3(3f, 0f, 5f), 20f, 0f, new float3(0f, 0f, 2f), gain)), 1);
                return (pulled.Inertial[0].position - plain.Inertial[0].position).xyz / dt2;
            };

            AssertNear(new float3(60f, 0f, 100f), pull(0f), 1e-4f, "constant stiffness: 20 N/m times the offset (3, 0, 5)");
            AssertNear(new float3(60f, 0f, 100f) * 2.5f, pull(0.5f), 1e-4f, "gain 0.5 per m at 3 m from the line: 1 + 1.5 = 2.5 times");

            // No line direction: no growth, whatever the gain.
            CpuBodyArrays plainBody = Scene(Box(new float3(0f), 1f, Wings));
            CpuBodyArrays noLine = Scene(Box(new float3(0f), 1f, Wings));
            CpuPredict.Run(plainBody, settings, 1);
            CpuPredict.Run(noLine, settings, 1, null, 0, Table(ForceFieldDescriptor.Spring(Wings, new float3(3f, 0f, 5f), 20f, 0f, float3.zero, 7f)), 1);
            AssertNear(new float3(60f, 0f, 100f), (noLine.Inertial[0].position - plainBody.Inertial[0].position).xyz / dt2, 1e-4f, "no line");
        }

        [Test]
        public void Spring_OfAnEmptyGroup_DoesNothing()
        {
            AvbdSolverSettings settings = Settings(float3.zero);
            CpuBodyArrays plain = Scene(Box(new float3(0f), 1f, Body));
            CpuBodyArrays bodies = Scene(Box(new float3(0f), 1f, Body));
            CpuPredict.Run(plain, settings, 1);
            CpuPredict.Run(bodies, settings, 1, null, 0, Table(ForceFieldDescriptor.Spring(Wings, new float3(0f, 6f, 0f), 100f)), 1);
            Assert.AreEqual(plain.Inertial[0].position, bodies.Inertial[0].position);
        }

        // ---- the GPU against the oracle ---------------------------------------------------------------------

        [Test]
        public void Gpu_Springs_MatchTheOracle_AlsoNextToOtherFields_AndAfterAnUpdate()
        {
            RequireCompute();
            var rng = new System.Random(77);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            {
                for (int i = 0; i < 90; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(F(rng, 0.3f, 2f), F(rng, 0.3f, 2f), F(rng, 0.3f, 2f)),
                        i % 9 == 0 ? 0f : F(rng, 0.2f, 2f), V(rng, 12f));
                    d.rotation = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f));
                    d.linearVelocity = V(rng, 10f);
                    d.kinematic = i % 13 == 0;
                    d.groupMask = (uint)rng.Next(0, 8); // three groups; some bodies are in none, some in several
                    buffers.Spawn(d);
                }
                buffers.ApplyEdits(applied);
                oracle.Apply(applied);

                var table = new ForceFieldRecord[AvbdSolver.MaxForceFields];
                using (var solver = new AvbdSolver(buffers, settings))
                {
                    ForceFieldDescriptor[] fields =
                    {
                        ForceFieldDescriptor.Spring(1u, new float3(3f, 8f, -2f), 150f, 1f),
                        ForceFieldDescriptor.Force(2u, new float3(5f, 0f, 1f)),
                        ForceFieldDescriptor.Spring(6u, new float3(-4f, 1f, 5f), 40f, 0.3f, new float3(0.2f, 0.3f, 1f), 0.8f), // overlaps the groups of the others
                    };
                    var added = new List<ForceFieldHandle>();
                    foreach (ForceFieldDescriptor f in fields)
                    {
                        added.Add(solver.AddForceField(f));
                        Assert.IsTrue(added[added.Count - 1].IsValid);
                        table[added.Count - 1] = f.ToRecord();
                    }

                    // Move the target of the first spring, as a hand that drags a group does, before the first predict.
                    ForceFieldDescriptor moved = ForceFieldDescriptor.Spring(1u, new float3(0f, 12f, 0f), 150f, 1f);
                    Assert.IsTrue(solver.UpdateForceField(added[0], moved));
                    table[0] = moved.ToRecord();

                    CpuPredict.Run(oracle, settings, buffers.HighWaterMark, null, 0, table, 3);
                    solver.DispatchPredict();
                }

                var state = new BodyStateRecord[Capacity];
                var inertial = new BodyInertialRecord[Capacity];
                buffers.State.GetData(state);
                buffers.Inertial.GetData(inertial);
                for (int slot = 0; slot < Capacity; slot++)
                {
                    string what = "slot " + slot;
                    AssertNear(oracle.Inertial[slot].position, inertial[slot].position, 1e-4f, what + " inertial position");
                    AssertNear(oracle.State[slot].position, state[slot].position, 1e-4f, what + " position");
                }
            }
        }

        [Test]
        public void Gpu_ASpringThatIsRemoved_NoLongerActs_AndOneWithoutBodiesDoesNothing()
        {
            RequireCompute();
            Func<int, float> run = mode => // 0: spring on the group, 1: spring removed again, 2: spring on a group without bodies
            {
                using (var buffers = new AvbdBodyBuffers(16))
                {
                    var settings = Settings(float3.zero);
                    using (var solver = new AvbdSolver(buffers, settings))
                    {
                        BodyDescriptor a = BodyDescriptor.Box(new float3(1f), 1f, new float3(0f));
                        a.groupMask = Wings;
                        BodyHandle ha = buffers.Spawn(a);
                        buffers.ApplyEdits();

                        ForceFieldHandle spring = solver.AddForceField(ForceFieldDescriptor.Spring(mode == 2 ? Body : Wings, new float3(0f, 6f, 0f), 100f, 0f));
                        if (mode == 1)
                            Assert.IsTrue(solver.RemoveForceField(spring));
                        solver.DispatchPredict();

                        var inertial = new BodyInertialRecord[16];
                        buffers.Inertial.GetData(inertial);
                        return inertial[ha.Slot].position.y;
                    }
                }
            };

            Assert.AreEqual(600f / 3600f, run(0), 1e-5f, "600 N for 1 kg over dt^2 = 1/3600");
            Assert.AreEqual(0f, run(1), 1e-6f);
            Assert.AreEqual(0f, run(2), 1e-6f);
        }

        [Test]
        public void Gpu_ForceFields_MatchTheOracle_OnRandomBodies_AlsoWithExplosionsInTheSameStep()
        {
            RequireCompute();
            var rng = new System.Random(2024);

            AvbdSolverSettings settings = AvbdSolverSettings.Default;
            var oracle = new CpuBodyArrays(Capacity);
            var applied = new List<EditCommand>();
            var handles = new List<BodyHandle>();

            using (var buffers = new AvbdBodyBuffers(Capacity))
            {
                for (int i = 0; i < 100; i++)
                {
                    BodyDescriptor d = BodyDescriptor.Box(new float3(F(rng, 0.3f, 2f), F(rng, 0.3f, 2f), F(rng, 0.3f, 2f)),
                        i % 7 == 0 ? 0f : F(rng, 0.2f, 2f), V(rng, 12f));
                    d.rotation = quaternion.EulerXYZ(F(rng, 0f, 6f), F(rng, 0f, 6f), F(rng, 0f, 6f));
                    d.linearVelocity = V(rng, 15f);
                    d.angularVelocity = V(rng, 5f);
                    d.kinematic = i % 11 == 0;
                    d.groupMask = (uint)rng.Next(0, 16); // four groups; some bodies are in none
                    handles.Add(buffers.Spawn(d));
                }
                for (int i = 3; i < handles.Count; i += 9)
                    buffers.Remove(handles[i]);

                buffers.ApplyEdits(applied);
                oracle.Apply(applied);
                for (int slot = 0; slot < buffers.HighWaterMark; slot++)
                    oracle.Motion[slot].prevLinearVel = new float4(oracle.Motion[slot].linearVel.xyz - V(rng, 0.2f), 0f);
                buffers.Motion.SetData(oracle.Motion);

                var table = new ForceFieldRecord[AvbdSolver.MaxForceFields];
                using (var solver = new AvbdSolver(buffers, settings))
                {
                    ForceFieldDescriptor[] fields =
                    {
                        ForceFieldDescriptor.Force(1u, new float3(30f, 5f, -20f)),
                        ForceFieldDescriptor.Force(2u, new float3(0f, 0f, 40f), ForceSpace.Body),
                        ForceFieldDescriptor.GravityScale(4u, 0.3f),
                        ForceFieldDescriptor.Drag(8u, 0.8f, 0.2f, new float3(3f, 0f, -2f)),
                        ForceFieldDescriptor.Wing(3u, new float3(0.2f, 1f, 0.1f), 1.5f, 4f, 0.04f, 1.2f, new float3(0f, 0f, 2f)),
                        ForceFieldDescriptor.Wing(8u, new float3(1f, 0f, 0f), 0.7f), // a field that is removed again below
                    };
                    var added = new List<ForceFieldHandle>();
                    foreach (ForceFieldDescriptor f in fields)
                    {
                        ForceFieldHandle h = solver.AddForceField(f);
                        Assert.IsTrue(h.IsValid);
                        table[added.Count] = f.ToRecord();
                        added.Add(h);
                    }

                    // Change one field and remove another while the table is already built: the shader must see the table as it is now.
                    ForceFieldDescriptor changed = ForceFieldDescriptor.GravityScale(4u, -0.5f);
                    Assert.IsTrue(solver.UpdateForceField(added[2], changed));
                    table[2] = changed.ToRecord();
                    Assert.IsTrue(solver.RemoveForceField(added[5]));
                    table[5] = default;
                    Assert.AreEqual(5, solver.ForceFieldCount);

                    var blasts = new[] { new CpuExplosion { center = new float3(1f, 2f, 3f), radius = 10f, impulse = 30f, seed = 0u } };
                    Assert.IsTrue(solver.Explode(blasts[0].center, blasts[0].radius, blasts[0].impulse));

                    CpuPredict.Run(oracle, settings, buffers.HighWaterMark, blasts, 1, table, 5);
                    solver.DispatchPredict();
                }

                var state = new BodyStateRecord[Capacity];
                var inertial = new BodyInertialRecord[Capacity];
                var motion = new BodyMotionRecord[Capacity];
                buffers.State.GetData(state);
                buffers.Inertial.GetData(inertial);
                buffers.Motion.GetData(motion);

                for (int slot = 0; slot < Capacity; slot++)
                {
                    string what = "slot " + slot;
                    AssertNear(oracle.Inertial[slot].position, inertial[slot].position, 1e-4f, what + " inertial position");
                    AssertNear(oracle.State[slot].position, state[slot].position, 1e-4f, what + " position");
                    AssertNear(oracle.State[slot].rotation, state[slot].rotation, 1e-5f, what + " rotation");
                    AssertNear(oracle.Motion[slot].linearVel, motion[slot].linearVel, 1e-4f, what + " velocity");
                }
            }
        }

        [Test]
        public void Gpu_WithoutFields_RunsThePlainKernel_AndAFieldChangesOnlyItsGroup()
        {
            RequireCompute();

            // Each case on fresh buffers: a predict moves the poses, so a second one would start from where the first left them. Two bodies at 10 m,
            // one in the wing group and one not; returns their inertial positions after one predict, with or without a gravity scale of 0 for the wings.
            Func<bool, float3[]> run = weightlessWings =>
            {
                using (var buffers = new AvbdBodyBuffers(16))
                using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default))
                {
                    BodyDescriptor a = BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f));
                    a.groupMask = Wings;
                    BodyDescriptor b = BodyDescriptor.Box(new float3(1f), 1f, new float3(5f, 10f, 0f));
                    b.groupMask = Body;
                    BodyHandle ha = buffers.Spawn(a);
                    BodyHandle hb = buffers.Spawn(b);
                    buffers.ApplyEdits();

                    if (weightlessWings)
                        Assert.IsTrue(solver.AddForceField(ForceFieldDescriptor.GravityScale(Wings, 0f)).IsValid);
                    solver.DispatchPredict();

                    var inertial = new BodyInertialRecord[16];
                    buffers.Inertial.GetData(inertial);
                    return new[] { inertial[ha.Slot].position.xyz, inertial[hb.Slot].position.xyz };
                }
            };

            float3[] plain = run(false);
            float3[] weightless = run(true);

            AssertNear(plain[0] - new float3(0f, 10f, 0f), plain[1] - new float3(5f, 10f, 0f), 1e-6f, "without fields both fall the same");
            Assert.Less(plain[0].y, 10f, "and they fall");
            Assert.AreEqual(10f, weightless[0].y, 1e-5f, "the wing group does not fall");
            AssertNear(plain[1], weightless[1], 1e-6f, "the other group falls as before");
        }

        [Test]
        public void Gpu_ARemovedField_NoLongerActs()
        {
            RequireCompute();
            Func<bool, float> run = removeIt =>
            {
                using (var buffers = new AvbdBodyBuffers(16))
                using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default))
                {
                    BodyDescriptor a = BodyDescriptor.Box(new float3(1f), 1f, new float3(0f, 10f, 0f));
                    a.groupMask = Wings;
                    BodyHandle ha = buffers.Spawn(a);
                    buffers.ApplyEdits();

                    ForceFieldHandle field = solver.AddForceField(ForceFieldDescriptor.GravityScale(Wings, 0f));
                    solver.DispatchPredict(); // uploads the table with the field in it; a weightless body at rest stays where it is
                    if (removeIt)
                        Assert.IsTrue(solver.RemoveForceField(field));
                    solver.DispatchPredict(); // must upload the table again

                    var inertial = new BodyInertialRecord[16];
                    buffers.Inertial.GetData(inertial);
                    return inertial[ha.Slot].position.y;
                }
            };

            Assert.AreEqual(10f, run(false), 1e-5f, "the field is still there");
            Assert.Less(run(true), 10f, "after the removal the body falls again");
        }

        // ---- the API ----------------------------------------------------------------------------------------

        [Test]
        public void AddUpdateRemove_TrackHandles_AndTheTableIsLimited()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default))
            {
                var handles = new List<ForceFieldHandle>();
                for (int i = 0; i < AvbdSolver.MaxForceFields; i++)
                {
                    ForceFieldHandle h = solver.AddForceField(ForceFieldDescriptor.Force(1u, new float3(i, 0f, 0f)));
                    Assert.IsTrue(h.IsValid);
                    handles.Add(h);
                }
                Assert.AreEqual(AvbdSolver.MaxForceFields, solver.ForceFieldCount);
                Assert.IsFalse(solver.AddForceField(ForceFieldDescriptor.Force(1u, float3.zero)).IsValid, "the table is full");

                Assert.IsTrue(solver.RemoveForceField(handles[3]));
                Assert.IsFalse(solver.RemoveForceField(handles[3]), "already removed");
                Assert.IsFalse(solver.IsForceFieldAlive(handles[3]));
                Assert.IsFalse(solver.UpdateForceField(handles[3], ForceFieldDescriptor.Force(1u, float3.zero)), "a stale handle changes nothing");
                Assert.AreEqual(AvbdSolver.MaxForceFields - 1, solver.ForceFieldCount);

                ForceFieldHandle reused = solver.AddForceField(ForceFieldDescriptor.GravityScale(2u, 0.5f));
                Assert.IsTrue(reused.IsValid);
                Assert.AreNotEqual(handles[3], reused, "the slot is reused but the handle is new");
                Assert.IsFalse(solver.IsForceFieldAlive(handles[3]), "the old handle stays stale");
                Assert.IsTrue(solver.IsForceFieldAlive(reused));

                Assert.IsFalse(solver.RemoveForceField(ForceFieldHandle.Invalid));
                Assert.IsFalse(ForceFieldHandle.Invalid.IsValid);
            }
        }

        [Test]
        public void AddForceField_RejectsBadValues()
        {
            RequireCompute();
            using (var buffers = new AvbdBodyBuffers(16))
            using (var solver = new AvbdSolver(buffers, AvbdSolverSettings.Default))
            {
                Assert.Throws<ArgumentException>(() => solver.AddForceField(default(ForceFieldDescriptor)), "no kind");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Force(1u, new float3(float.NaN, 0f, 0f))));
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.GravityScale(1u, float.PositiveInfinity)));
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Drag(1u, -1f, 0f)));
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Drag(1u, 0f, 0f, new float3(0f, float.NaN, 0f))));
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Wing(1u, float3.zero, 1f)), "zero normal");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Wing(1u, new float3(0f, 1f, 0f), 0f)), "no area");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Wing(1u, new float3(0f, 1f, 0f), 1f, -1f)), "negative slope");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Wing(1u, new float3(0f, 1f, 0f), 1f, 4f, 0f, 0f)), "no air");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Spring(1u, new float3(0f, float.NaN, 0f), 10f)), "target");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Spring(1u, float3.zero, -10f)), "negative stiffness");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Spring(1u, float3.zero, 10f, -1f)), "negative damping");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Spring(1u, float3.zero, 10f, 1f, float3.zero, -1f)), "negative gain");
                Assert.Throws<ArgumentException>(() => solver.AddForceField(ForceFieldDescriptor.Spring(1u, float3.zero, 10f, 1f, new float3(0f, float.NaN, 1f), 1f)), "line");
                Assert.AreEqual(0, solver.ForceFieldCount, "nothing was added");

                ForceFieldHandle h = solver.AddForceField(ForceFieldDescriptor.GravityScale(1u, 0.5f));
                Assert.Throws<ArgumentException>(() => solver.UpdateForceField(h, ForceFieldDescriptor.Wing(1u, float3.zero, 1f)));
                Assert.IsTrue(solver.IsForceFieldAlive(h), "a bad update keeps the field as it was");
            }
        }

        // ---- helpers ----------------------------------------------------------------------------------------

        static AvbdSolverSettings Settings(float3 gravity)
        {
            AvbdSolverSettings s = AvbdSolverSettings.Default;
            s.gravity = gravity;
            return s;
        }

        // A 1 m box of the given density (mass = density, in kg) with a group mask.
        static BodyDescriptor Box(float3 position, float density, uint groupMask)
        {
            BodyDescriptor d = BodyDescriptor.Box(new float3(1f), density, position);
            d.groupMask = groupMask;
            return d;
        }

        static BodyDescriptor Kinematic(float3 position, uint groupMask)
        {
            BodyDescriptor d = Box(position, 1f, groupMask);
            d.kinematic = true;
            return d;
        }

        static BodyDescriptor Static(float3 position, uint groupMask) => Box(position, 0f, groupMask);

        static CpuBodyArrays Scene(params BodyDescriptor[] descriptors)
        {
            var queue = new BodyEditQueue(16);
            var oracle = new CpuBodyArrays(16);
            foreach (BodyDescriptor d in descriptors)
                queue.Spawn(d);
            var commands = new List<EditCommand>();
            queue.Drain(commands);
            oracle.Apply(commands);
            return oracle;
        }

        // The table of the solver after adding the fields in order.
        static ForceFieldRecord[] Table(params ForceFieldDescriptor[] fields)
        {
            var table = new ForceFieldRecord[AvbdSolver.MaxForceFields];
            for (int i = 0; i < fields.Length; i++)
                table[i] = fields[i].ToRecord();
            return table;
        }

        static void RequireCompute()
        {
            if (!SystemInfo.supportsComputeShaders)
                Assert.Ignore("Compute shaders are not supported on this device.");
        }

        static float F(System.Random rng, float min, float max) => min + (float)rng.NextDouble() * (max - min);

        static float3 V(System.Random rng, float range) => new float3(F(rng, -range, range), F(rng, -range, range), F(rng, -range, range));

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
