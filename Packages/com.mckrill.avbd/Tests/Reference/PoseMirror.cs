using NUnit.Framework;

namespace McKrill.Avbd.Tests
{
    /// <summary>
    /// The compact pose array (<see cref="AvbdBodyBuffers.Pose"/>) is a copy of the pose in the state array, which every kernel that writes the pose keeps in
    /// step. The primal and dual stages read the neighbors of joints and springs from the copy, so a kernel that forgets to write it makes them solve with
    /// a stale pose. This check is called after the stages that write poses.
    /// </summary>
    static class PoseMirror
    {
        public static void Assert(AvbdBodyBuffers buffers, string when)
        {
            var state = new BodyStateRecord[buffers.Capacity];
            var pose = new BodyPoseRecord[buffers.Capacity];
            buffers.State.GetData(state);
            buffers.Pose.GetData(pose);
            for (int i = 0; i < state.Length; i++)
            {
                if (!state[i].position.Equals(pose[i].position) || !state[i].rotation.Equals(pose[i].rotation))
                    NUnit.Framework.Assert.Fail("The pose array differs from the state array in slot " + i + " " + when + ".\n  state: " + state[i].position + " " + state[i].rotation
                                                + "\n  pose:  " + pose[i].position + " " + pose[i].rotation);
            }
        }
    }
}
