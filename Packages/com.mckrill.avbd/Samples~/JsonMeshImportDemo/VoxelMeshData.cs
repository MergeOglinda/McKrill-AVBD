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
    ///
    /// A file written with <c>--smooth</c> also has the optional chunk arrays: for each of the <see cref="chunkCount"/> voxels on the surface of the object, the
    /// piece of the smooth surface that belongs to that block, as a closed triangle mesh in the frame of the voxel (<see cref="HasChunks"/>).
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

        // Optional (the voxelizer's --smooth): chunk c belongs to voxel chunkVoxels[c], has the vertices chunkVertexStart[c] to chunkVertexStart[c + 1] - 1 (three
        // floats each in chunkPositions and chunkNormals, relative to the center of the voxel, in the units of the file) and the triangles chunkTriangleStart[c]
        // to chunkTriangleStart[c + 1] - 1 (three indices each in chunkTriangles, counted from the chunk's first vertex).
        public int chunkCount;
        public int[] chunkVoxels;
        public int[] chunkVertexStart;
        public int[] chunkTriangleStart;
        public float[] chunkPositions;
        public float[] chunkNormals;
        public int[] chunkTriangles;

        /// <summary>True when the file carries the smooth surface of the blocks on the surface of the object.</summary>
        public bool HasChunks => chunkCount > 0;

        /// <summary>The total number of vertices of all chunks.</summary>
        public int ChunkVertexCount => HasChunks ? chunkVertexStart[chunkCount] : 0;

        /// <summary>The total number of triangles of all chunks.</summary>
        public int ChunkTriangleCount => HasChunks ? chunkTriangleStart[chunkCount] : 0;

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
            return ValidateChunks();
        }

        // The chunk arrays are optional. A file that has some of them but not all, or inconsistent ones, is an error (it is not silently shown as cubes).
        string ValidateChunks()
        {
            bool any = chunkCount != 0 || NotEmpty(chunkVoxels) || NotEmpty(chunkVertexStart) || NotEmpty(chunkTriangleStart) || NotEmpty(chunkPositions)
                       || NotEmpty(chunkNormals) || NotEmpty(chunkTriangles);
            if (!any)
                return null;
            if (chunkCount <= 0)
                return "the chunk arrays are there but chunkCount is not positive";
            if (chunkVoxels == null || chunkVoxels.Length != chunkCount)
                return "chunkVoxels does not have chunkCount (" + chunkCount + ") integers";
            if (chunkVertexStart == null || chunkVertexStart.Length != chunkCount + 1 || chunkTriangleStart == null || chunkTriangleStart.Length != chunkCount + 1)
                return "chunkVertexStart and chunkTriangleStart do not have chunkCount + 1 (" + (chunkCount + 1) + ") integers";
            if (chunkVertexStart[0] != 0 || chunkTriangleStart[0] != 0)
                return "the chunk start arrays do not begin with 0";
            for (int c = 0; c < chunkCount; c++)
            {
                if (chunkVoxels[c] < 0 || chunkVoxels[c] >= voxelCount)
                    return "chunk " + c + " belongs to a voxel that does not exist";
                if (c > 0 && chunkVoxels[c] <= chunkVoxels[c - 1])
                    return "chunkVoxels is not increasing (a voxel has one chunk)";
                if (chunkVertexStart[c + 1] <= chunkVertexStart[c] || chunkTriangleStart[c + 1] <= chunkTriangleStart[c])
                    return "chunk " + c + " is empty or its start numbers decrease";
            }
            int vertices = chunkVertexStart[chunkCount];
            int triangles = chunkTriangleStart[chunkCount];
            if (chunkPositions == null || chunkPositions.Length != 3 * vertices || chunkNormals == null || chunkNormals.Length != 3 * vertices)
                return "chunkPositions and chunkNormals do not have three floats for each of the " + vertices + " vertices";
            if (chunkTriangles == null || chunkTriangles.Length != 3 * triangles)
                return "chunkTriangles does not have three integers for each of the " + triangles + " triangles";
            for (int c = 0; c < chunkCount; c++)
            {
                int count = chunkVertexStart[c + 1] - chunkVertexStart[c];
                for (int m = 3 * chunkTriangleStart[c]; m < 3 * chunkTriangleStart[c + 1]; m++)
                    if (chunkTriangles[m] < 0 || chunkTriangles[m] >= count)
                        return "a triangle of chunk " + c + " refers to a vertex outside of the chunk";
            }
            return null;
        }

        static bool NotEmpty(Array array)
        {
            return array != null && array.Length > 0;
        }
    }
}
