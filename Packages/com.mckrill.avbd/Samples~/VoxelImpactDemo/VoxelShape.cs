using System;
using Unity.Mathematics;

namespace McKrill.Avbd.Samples
{
    /// <summary>
    /// A solid shape on a grid of voxels (a cube, a sphere or a torus), optionally hollowed out to a skin, with the joints that link its voxels: the
    /// stand-in for a voxel file of the JSON Mesh Import Demo (<c>Tools/Voxelize</c>) that needs no file. Pure C#, no GPU.
    ///
    /// All sizes are in voxels. The grid is as big as the shape plus one empty layer on every side, a cell (i, j, k) has its center at
    /// (i + 0.5 - nx / 2, j + 0.5 - ny / 2, k + 0.5 - nz / 2) voxels from the middle of the shape, and a cell is inside if its center is. The
    /// offsets are multiples of 0.5, so the doubles are exact and the shapes are the same on every machine.
    /// The cells are numbered in the order of the loops over i, then j, then k. A voxel is linked to the voxels in its 18-neighborhood (those that share a
    /// face or an edge with it), one joint for each pair, like the voxelizer's <c>--connectivity 18</c>.
    ///
    /// The same file is in the performance tests (<c>Tests/Performance/VoxelShape.cs</c>), apart from the namespace: keep the two the same.
    /// </summary>
    internal sealed class VoxelShape
    {
        public readonly int nx, ny, nz;
        readonly bool[] inside;   // (i * ny + j) * nz + k

        /// <summary>The cells that are inside, in the order of their numbers.</summary>
        public readonly int3[] cells;

        // Number of the cell at a grid index, or -1 if the cell is empty.
        readonly int[] numbers;

        // The half of the 18-neighborhood that is "after" a cell: the offsets with one or two non-zero parts whose first non-zero part is positive.
        static readonly int3[] HalfOffsets = MakeHalfOffsets();

        VoxelShape(int nx, int ny, int nz, bool[] inside)
        {
            this.nx = nx;
            this.ny = ny;
            this.nz = nz;
            this.inside = inside;
            numbers = new int[inside.Length];
            int count = 0;
            for (int n = 0; n < inside.Length; n++)
                numbers[n] = inside[n] ? count++ : -1;
            cells = new int3[count];
            count = 0;
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                        if (inside[(i * ny + j) * nz + k])
                            cells[count++] = new int3(i, j, k);
        }

        /// <summary>Number of voxels.</summary>
        public int Count => cells.Length;

        public bool Contains(int i, int j, int k)
        {
            return i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz && inside[(i * ny + j) * nz + k];
        }

        /// <summary>Position of the center of cell n, in voxels from the middle of the shape.</summary>
        public float3 CellOffset(int n)
        {
            int3 c = cells[n];
            return new float3(c.x + 0.5f - nx * 0.5f, c.y + 0.5f - ny * 0.5f, c.z + 0.5f - nz * 0.5f);
        }

        /// <summary>A cube of <paramref name="edge"/> voxels.</summary>
        public static VoxelShape Cube(int edge)
        {
            if (edge < 1) throw new ArgumentOutOfRangeException(nameof(edge));
            int n = edge + 2;
            var inside = new bool[n * n * n];
            for (int i = 1; i <= edge; i++)
                for (int j = 1; j <= edge; j++)
                    for (int k = 1; k <= edge; k++)
                        inside[(i * n + j) * n + k] = true;
            return new VoxelShape(n, n, n, inside);
        }

        /// <summary>A ball of <paramref name="radius"/> voxels.</summary>
        public static VoxelShape Sphere(int radius)
        {
            if (radius < 1) throw new ArgumentOutOfRangeException(nameof(radius));
            int n = 2 * radius + 2;
            double r2 = (double)radius * radius;
            var inside = new bool[n * n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < n; k++)
                    {
                        double x = i + 0.5 - n / 2.0, y = j + 0.5 - n / 2.0, z = k + 0.5 - n / 2.0;
                        inside[(i * n + j) * n + k] = x * x + y * y + z * z <= r2;
                    }
            return new VoxelShape(n, n, n, inside);
        }

        /// <summary>
        /// A torus around the z axis: the circle of radius <paramref name="major"/> voxels in the x-y plane, thickened to a tube of radius
        /// <paramref name="minor"/> voxels.
        /// </summary>
        public static VoxelShape Torus(int major, int minor)
        {
            if (minor < 1 || major <= minor) throw new ArgumentOutOfRangeException(nameof(major), "a torus needs 0 < minor < major");
            int n = 2 * (major + minor) + 2;
            int m = 2 * minor + 2;
            double r2 = (double)minor * minor;
            var inside = new bool[n * n * m];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    for (int k = 0; k < m; k++)
                    {
                        double x = i + 0.5 - n / 2.0, y = j + 0.5 - n / 2.0, z = k + 0.5 - m / 2.0;
                        double d = Math.Sqrt(x * x + y * y) - major;
                        inside[(i * n + j) * m + k] = d * d + z * z <= r2;
                    }
            return new VoxelShape(n, n, m, inside);
        }

        /// <summary>
        /// The skin of the shape, <paramref name="layers"/> voxels thick: the voxels that are removed by <paramref name="layers"/> rounds of erosion (a voxel
        /// stays only if all the 26 voxels around it are inside) are what is left out.
        /// </summary>
        public VoxelShape Shell(int layers)
        {
            bool[] core = (bool[])inside.Clone();
            for (int l = 0; l < layers; l++)
                core = Erode(core);
            var skin = new bool[inside.Length];
            for (int n = 0; n < skin.Length; n++)
                skin[n] = inside[n] && !core[n];
            return new VoxelShape(nx, ny, nz, skin);
        }

        bool[] Erode(bool[] mask)
        {
            var result = new bool[mask.Length];
            for (int i = 0; i < nx; i++)
                for (int j = 0; j < ny; j++)
                    for (int k = 0; k < nz; k++)
                    {
                        bool all = true;
                        for (int di = -1; di <= 1 && all; di++)
                            for (int dj = -1; dj <= 1 && all; dj++)
                                for (int dk = -1; dk <= 1 && all; dk++)
                                {
                                    int a = i + di, b = j + dj, c = k + dk;
                                    all = a >= 0 && a < nx && b >= 0 && b < ny && c >= 0 && c < nz && mask[(a * ny + b) * nz + c];
                                }
                        result[(i * ny + j) * nz + k] = all;
                    }
            return result;
        }

        /// <summary>Number of joints of <see cref="Pairs"/>.</summary>
        public int CountPairs()
        {
            int total = 0;
            for (int n = 0; n < cells.Length; n++)
                for (int o = 0; o < HalfOffsets.Length; o++)
                {
                    int3 c = cells[n] + HalfOffsets[o];
                    if (Contains(c.x, c.y, c.z))
                        total++;
                }
            return total;
        }

        /// <summary>The links: the numbers of the two voxels of each pair, one pair after the other (2 * <see cref="CountPairs"/> numbers).</summary>
        public int[] Pairs()
        {
            var pairs = new int[2 * CountPairs()];
            int p = 0;
            for (int n = 0; n < cells.Length; n++)
                for (int o = 0; o < HalfOffsets.Length; o++)
                {
                    int3 c = cells[n] + HalfOffsets[o];
                    if (!Contains(c.x, c.y, c.z))
                        continue;
                    pairs[p++] = n;
                    pairs[p++] = numbers[(c.x * ny + c.y) * nz + c.z];
                }
            return pairs;
        }

        static int3[] MakeHalfOffsets()
        {
            var list = new System.Collections.Generic.List<int3>();
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int parts = (dx != 0 ? 1 : 0) + (dy != 0 ? 1 : 0) + (dz != 0 ? 1 : 0);
                        bool after = dx > 0 || (dx == 0 && (dy > 0 || (dy == 0 && dz > 0)));
                        if (parts >= 1 && parts <= 2 && after)
                            list.Add(new int3(dx, dy, dz));
                    }
            return list.ToArray();
        }
    }
}
