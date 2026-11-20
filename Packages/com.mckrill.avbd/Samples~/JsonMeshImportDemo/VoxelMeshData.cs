using System;
using System.IO;
using Unity.Mathematics;
using UnityEngine;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// The contents of a voxel file written by <c>Tools/Voxelize/voxelize_stl.py</c> (format <c>mckrill-avbd-voxels</c>, version 1; the file format is
    /// described in <c>Tools/Voxelize/README.md</c>). The field names are the JSON keys, so <see cref="JsonUtility"/> fills the object directly.
    ///
    /// Voxel <c>n</c> is the box of edge <see cref="voxelSize"/> centered at <c>origin + (i + 0.5, j + 0.5, k + 0.5) * voxelSize</c>, where <c>(i, j, k)</c> are
    /// the three integers at <c>voxels[3 * n]</c>. A joint is a pair of voxel numbers at <c>joints[2 * m]</c>.
    /// </summary>
    [Serializable]
    public sealed class VoxelMeshData
    {
        public const string FormatName = "mckrill-avbd-voxels";
        public const int SupportedVersion = 1;

        public string format;
        public int version;
        public string source;
        public float voxelSize;
        public float[] origin;
        public int[] gridDimensions;
        public int connectivity;
        public string surface;
        public int voxelCount;
        public int jointCount;
        public int[] voxels;
        public int[] joints;

        /// <summary>Center of voxel <paramref name="number"/> in the file's own frame.</summary>
        public float3 VoxelCenter(int number)
        {
            return new float3(
                origin[0] + (voxels[3 * number] + 0.5f) * voxelSize,
                origin[1] + (voxels[3 * number + 1] + 0.5f) * voxelSize,
                origin[2] + (voxels[3 * number + 2] + 0.5f) * voxelSize);
        }

        /// <summary>
        /// Reads a voxel file. <paramref name="entry"/> is either the path of a file on disk (when it is rooted, for example <c>C:/models/engine.json</c>) or the name of a
        /// <see cref="TextAsset"/> in a <c>Resources</c> folder, with or without the <c>.json</c> extension (for example <c>JsonMeshImportDemo/Sphere</c>).
        /// </summary>
        public static bool TryLoad(string entry, out VoxelMeshData data, out string error)
        {
            data = null;
            string json;
            if (string.IsNullOrWhiteSpace(entry))
            {
                error = "the file name is empty";
                return false;
            }

            if (Path.IsPathRooted(entry))
            {
                try
                {
                    json = File.ReadAllText(entry);
                }
                catch (Exception e) when (e is IOException || e is UnauthorizedAccessException)
                {
                    error = "cannot read the file: " + e.Message;
                    return false;
                }
            }
            else
            {
                string resourceName = entry.EndsWith(".json", StringComparison.OrdinalIgnoreCase) ? entry.Substring(0, entry.Length - ".json".Length) : entry;
                var asset = Resources.Load<TextAsset>(resourceName);
                if (asset == null)
                {
                    error = "there is no text asset \"" + resourceName + "\" in a Resources folder (a file outside the project is given by its full path)";
                    return false;
                }
                json = asset.text;
            }

            return TryParse(json, out data, out error);
        }

        /// <summary>Parses the JSON text of a voxel file and checks that the arrays are consistent, so that the demo can index them without further tests.</summary>
        public static bool TryParse(string json, out VoxelMeshData data, out string error)
        {
            data = null;
            VoxelMeshData parsed;
            try
            {
                parsed = JsonUtility.FromJson<VoxelMeshData>(json);
            }
            catch (ArgumentException e)
            {
                error = "not valid JSON: " + e.Message;
                return false;
            }

            error = parsed == null ? "the file is empty" : parsed.Validate();
            if (error != null)
                return false;
            data = parsed;
            return true;
        }

        // Returns null when the data is usable, else what is wrong.
        string Validate()
        {
            if (format != FormatName)
                return "the format is \"" + format + "\", not \"" + FormatName + "\"";
            if (version != SupportedVersion)
                return "format version " + version + " is not supported (this demo reads version " + SupportedVersion + ")";
            if (!(voxelSize > 0f) || float.IsInfinity(voxelSize))
                return "the voxel size is not a positive number";
            if (origin == null || origin.Length != 3)
                return "the origin does not have three numbers";
            if (gridDimensions == null || gridDimensions.Length != 3 || gridDimensions[0] <= 0 || gridDimensions[1] <= 0 || gridDimensions[2] <= 0)
                return "the grid dimensions are not three positive integers";
            if (voxelCount <= 0)
                return "the file has no voxels";
            if (voxels == null || voxels.Length != 3 * voxelCount)
                return "the voxel array does not have 3 * voxelCount (" + 3 * voxelCount + ") integers";
            if (jointCount < 0 || joints == null || joints.Length != 2 * jointCount)
                return "the joint array does not have 2 * jointCount (" + 2 * jointCount + ") integers";

            for (int n = 0; n < voxelCount; n++)
                for (int axis = 0; axis < 3; axis++)
                {
                    int index = voxels[3 * n + axis];
                    if (index < 0 || index >= gridDimensions[axis])
                        return "voxel " + n + " lies outside the grid";
                }
            for (int m = 0; m < 2 * jointCount; m++)
                if (joints[m] < 0 || joints[m] >= voxelCount)
                    return "joint " + m / 2 + " refers to a voxel that does not exist";
            return null;
        }
    }
}
