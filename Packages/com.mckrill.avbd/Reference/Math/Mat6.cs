using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// A 6x6 matrix stored as four 3x3 blocks. Rows/columns 0-2 are the linear degrees of freedom of a
    /// rigid body and 3-5 the angular ones. Used for the per-body Hessian of the AVBD primal update,
    /// which is symmetric positive definite and solved with <see cref="TrySolveLDLT"/>.
    /// Written for clarity and determinism (the CPU oracle), not for speed.
    /// </summary>
    public struct Mat6
    {
        public float3x3 ll; // rows 0-2, cols 0-2
        public float3x3 la; // rows 0-2, cols 3-5
        public float3x3 al; // rows 3-5, cols 0-2
        public float3x3 aa; // rows 3-5, cols 3-5

        public static Mat6 Zero => default;

        public static Mat6 Identity => new Mat6 { ll = float3x3.identity, aa = float3x3.identity };

        // ---- element access ------------------------------------------------------------------

        public float Get(int r, int c)
        {
            if (r < 3) return c < 3 ? ll[c][r] : la[c - 3][r];
            return c < 3 ? al[c][r - 3] : aa[c - 3][r - 3];
        }

        public void Set(int r, int c, float v)
        {
            if (r < 3)
            {
                if (c < 3) SetEntry(ref ll, r, c, v);
                else SetEntry(ref la, r, c - 3, v);
            }
            else
            {
                if (c < 3) SetEntry(ref al, r - 3, c, v);
                else SetEntry(ref aa, r - 3, c - 3, v);
            }
        }

        static void SetEntry(ref float3x3 m, int r, int c, float v)
        {
            float3 col = m[c];
            col[r] = v;
            m[c] = col;
        }

        // ---- assembly helpers (keep the matrix symmetric) ---------------------------------------

        /// <summary>Adds a 3x3 block to the linear-linear corner.</summary>
        public void AddLinear(float3x3 b) => ll += b;

        /// <summary>Adds a 3x3 block to the angular-angular corner.</summary>
        public void AddAngular(float3x3 b) => aa += b;

        /// <summary>Adds b to the linear-angular block and transpose(b) to the angular-linear block.</summary>
        public void AddCoupling(float3x3 b)
        {
            la += b;
            al += math.transpose(b);
        }

        /// <summary>Adds lin * I to the linear diagonal and ang * I to the angular diagonal.</summary>
        public void AddScaledIdentity(float lin, float ang)
        {
            ll += lin * float3x3.identity;
            aa += ang * float3x3.identity;
        }

        /// <summary>Adds k * j * j^T (a rank-one, stiffness-weighted Jacobian term).</summary>
        public void AddOuter(Vec6 j, float k)
        {
            ll += k * AvbdQuat.Outer(j.lin, j.lin);
            aa += k * AvbdQuat.Outer(j.ang, j.ang);
            AddCoupling(k * AvbdQuat.Outer(j.lin, j.ang));
        }

        /// <summary>Adds another matrix element-wise.</summary>
        public void Add(Mat6 o)
        {
            ll += o.ll;
            la += o.la;
            al += o.al;
            aa += o.aa;
        }

        // ---- products ---------------------------------------------------------------------------

        public Vec6 Mul(Vec6 v)
        {
            return new Vec6(
                math.mul(ll, v.lin) + math.mul(la, v.ang),
                math.mul(al, v.lin) + math.mul(aa, v.ang));
        }

        public bool IsSymmetric(float tolerance)
        {
            for (int r = 0; r < 6; r++)
                for (int c = r + 1; c < 6; c++)
                    if (math.abs(Get(r, c) - Get(c, r)) > tolerance) return false;
            return true;
        }

        // ---- LDL^T solve ------------------------------------------------------------------------

        /// <summary>
        /// Solves A x = b for a symmetric positive definite A using an LDL^T factorization (no pivoting).
        /// Only the lower triangle of A is read. Returns false, with x = 0, if a pivot is not greater than
        /// <paramref name="minPivot"/> or is NaN, i.e. when A is not numerically positive definite.
        /// </summary>
        public bool TrySolveLDLT(Vec6 b, out Vec6 x, float minPivot = 1e-20f)
        {
            Mat6 a = this; // factored in place: strict lower triangle = L (unit diagonal implied), diagonal = D
            Vec6 v = default;

            for (int j = 0; j < 6; j++)
            {
                for (int k = 0; k < j; k++) v[k] = a.Get(j, k) * a.Get(k, k);

                float dj = a.Get(j, j);
                for (int k = 0; k < j; k++) dj -= a.Get(j, k) * v[k];

                if (!(dj > minPivot)) // also rejects NaN
                {
                    x = default;
                    return false;
                }
                a.Set(j, j, dj);

                float inv = 1f / dj;
                for (int i = j + 1; i < 6; i++)
                {
                    float s = a.Get(i, j);
                    for (int k = 0; k < j; k++) s -= a.Get(i, k) * v[k];
                    a.Set(i, j, s * inv);
                }
            }

            Vec6 y = b;

            // Forward substitution: L y = b
            for (int i = 0; i < 6; i++)
            {
                float s = y[i];
                for (int k = 0; k < i; k++) s -= a.Get(i, k) * y[k];
                y[i] = s;
            }

            // Diagonal: D z = y
            for (int i = 0; i < 6; i++) y[i] /= a.Get(i, i);

            // Backward substitution: L^T x = z
            for (int i = 5; i >= 0; i--)
            {
                float s = y[i];
                for (int k = i + 1; k < 6; k++) s -= a.Get(k, i) * y[k];
                y[i] = s;
            }

            x = y;
            return true;
        }
    }
}
