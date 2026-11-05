"""
Literal numpy (float32) mirror of Packages/com.mckrill.avbd/Reference/Math/*.cs, used to validate the ALGORITHMS when no
C#/Unity toolchain is available. It does not prove the C# compiles; it proves the math and the
index/loop structure the C# uses are correct. Quaternions use Unity layout: (x, y, z, w).

Run:  python3 -I Tools/verify_reference_math.py
"""
import sys
import numpy as np

f32 = np.float32
rng = np.random.default_rng(12345)
failures = []


def check(cond, name):
    if not cond:
        failures.append(name)
        print(f"FAIL  {name}")


# ---- quaternion helpers (mirror of AvbdQuat.cs) ---------------------------------------------------

def qmul(a, b):  # Hamilton product, Unity.Mathematics math.mul(quaternion, quaternion)
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=f32)


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=f32)


def qnorm(q):
    return (q / np.linalg.norm(q)).astype(f32)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=f32)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(f32(angle) * f32(0.5))
    return np.array([*(axis * s), np.cos(f32(angle) * f32(0.5))], dtype=f32)


def rotate(q, v):
    qv = np.array([*v, 0.0], dtype=f32)
    return qmul(qmul(q, qv), qconj(q))[:3]


def rot_matrix(q):  # float3x3(quaternion)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=f32)


def difference(a, b):
    d = qmul(a, qconj(b))
    s = f32(-2.0) if d[3] < 0 else f32(2.0)
    return (s * d[:3]).astype(f32)


def integrate_angular(q, dw):
    dq = np.array([*dw, 0.0], dtype=f32)
    r = q + f32(0.5) * qmul(dq, q)
    return qnorm(r)


def rotated_diagonal(q, d):
    r = rot_matrix(q)
    return (r @ np.diag(np.asarray(d, dtype=f32)) @ r.T).astype(f32)


def outer(a, b):
    return np.outer(a, b).astype(f32)


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=f32)


# ---- Mat6 / LDL^T (mirror of Mat6.cs, same loop structure) -----------------------------------------

def try_solve_ldlt(a_in, b, min_pivot=1e-20):
    a = a_in.copy().astype(f32)
    v = np.zeros(6, dtype=f32)
    for j in range(6):
        for k in range(j):
            v[k] = a[j, k] * a[k, k]
        dj = a[j, j]
        for k in range(j):
            dj = f32(dj - a[j, k] * v[k])
        if not (dj > min_pivot):
            return False, np.zeros(6, dtype=f32)
        a[j, j] = dj
        inv = f32(1.0) / dj
        for i in range(j + 1, 6):
            s = a[i, j]
            for k in range(j):
                s = f32(s - a[i, k] * v[k])
            a[i, j] = f32(s * inv)
    y = b.copy().astype(f32)
    for i in range(6):
        s = y[i]
        for k in range(i):
            s = f32(s - a[i, k] * y[k])
        y[i] = s
    for i in range(6):
        y[i] = f32(y[i] / a[i, i])
    for i in range(5, -1, -1):
        s = y[i]
        for k in range(i + 1, 6):
            s = f32(s - a[k, i] * y[k])
        y[i] = s
    return True, y


def random_spd(shift=1.0):
    b = rng.uniform(-1, 1, (6, 6)).astype(f32)
    return (b @ b.T + f32(shift) * np.eye(6, dtype=f32)).astype(f32)


# ---- tests (mirror of ReferenceMathTests.cs) ---------------------------------------------------------

# LDLT solves random SPD systems; also compare with a float64 reference solve.
worst = 0.0
for trial in range(200):
    a = random_spd()
    x_true = rng.uniform(-3, 3, 6).astype(f32)
    b = (a @ x_true).astype(f32)
    ok, x = try_solve_ldlt(a, b)
    check(ok, f"ldlt solve ok trial {trial}")
    ref = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
    worst = max(worst, float(np.linalg.norm(x - ref)))
    check(np.linalg.norm(a @ x - b) < 1e-3, f"ldlt residual trial {trial}")
    check(np.linalg.norm(x - x_true) < 1e-3, f"ldlt error trial {trial}")
print(f"LDLT: worst |x - float64 reference| over 200 SPD systems = {worst:.2e}")

# Diagonal system is exact.
a = np.diag(np.arange(1, 7, dtype=f32))
ok, x = try_solve_ldlt(a, np.arange(1, 7, dtype=f32))
check(ok and np.allclose(x, 1.0, atol=1e-6), "ldlt diagonal exact")

# Rejects zero / negative / NaN.
check(not try_solve_ldlt(np.zeros((6, 6), dtype=f32), np.zeros(6, dtype=f32))[0], "ldlt rejects zero")
neg = np.eye(6, dtype=f32); neg[3, 3] = -1
check(not try_solve_ldlt(neg, np.zeros(6, dtype=f32))[0], "ldlt rejects negative")
nan = np.eye(6, dtype=f32); nan[2, 2] = np.nan
check(not try_solve_ldlt(nan, np.zeros(6, dtype=f32))[0], "ldlt rejects NaN")

# Reads only the lower triangle.
a = random_spd(); b = rng.uniform(-2, 2, 6).astype(f32)
c = a.copy()
c[np.triu_indices(6, 1)] = 999
ok1, x1 = try_solve_ldlt(a, b)
ok2, x2 = try_solve_ldlt(c, b)
check(ok1 and ok2 and np.array_equal(x1, x2), "ldlt lower triangle only")

# Rigid-body-shaped Hessian (large entries): relative residual.
dt = f32(1 / 60)
q = axis_angle([1, 2, 3], 0.7)
h = np.zeros((6, 6), dtype=f32)
h[:3, :3] += (f32(2.0) / (dt * dt)) * np.eye(3, dtype=f32)
h[3:, 3:] += rotated_diagonal(q, [0.5, 1, 2]) / (dt * dt)
j = np.array([0, 1, 0, 0.3, 0, -0.2], dtype=f32)
h += f32(1000) * np.outer(j, j).astype(f32)
f = np.array([1, -2, 0.5, 0.1, 0.2, -0.3], dtype=f32)
ok, dx = try_solve_ldlt(h, f)
check(ok and np.linalg.norm(h @ dx - f) / np.linalg.norm(f) < 1e-4, "ldlt rigid-body hessian")

# Quaternion tests.
q = axis_angle([0.3, -0.5, 0.8], 1.1)
check(np.linalg.norm(difference(q, q)) < 1e-5, "difference of equal is zero")

axis = np.array([1, 2, -1], dtype=f32); axis /= np.linalg.norm(axis)
a = axis_angle(axis, 0.01)
check(np.allclose(difference(a, np.array([0, 0, 0, 1], dtype=f32)), axis * f32(0.01), atol=1e-6),
      "difference small rotation ~ rotation vector")

a = axis_angle([0, 1, 1], 0.4); b = axis_angle([1, 0, 1], -0.3)
neg_a = -a
check(np.allclose(difference(a, b), difference(neg_a, b), atol=1e-5), "difference sign invariant")

q = np.array([0, 0, 0, 1], dtype=f32)
unit_ok = True
for i in range(1000):
    q = integrate_angular(q, rng.uniform(-0.2, 0.2, 3).astype(f32))
    unit_ok &= abs(np.linalg.norm(q) - 1) < 1e-5
check(unit_ok, "integrate_angular keeps unit length (1000 steps)")

q = axis_angle([1, 1, 0], 0.9)
dw = np.array([0.004, -0.006, 0.003], dtype=f32)
approx = integrate_angular(q, dw)
exact = qmul(axis_angle(dw, float(np.linalg.norm(dw))), q)
err = np.linalg.norm(difference(approx, exact))
check(err < 1e-4, "integrate_angular matches exact small rotation")
print(f"integrate_angular small-delta error vs exact: {err:.2e}")

q = axis_angle([-1, 0.5, 2], 2.0)
dw = np.array([0.01, 0.02, -0.015], dtype=f32)
check(np.allclose(difference(integrate_angular(q, dw), q), dw, atol=1e-4), "integrate then difference recovers delta")

q = axis_angle([1, -1, 2], 1.3)
d = np.array([0.5, 1.5, 3], dtype=f32)
m = rotated_diagonal(q, d)
check(np.allclose(m, m.T, atol=1e-5) and abs(np.trace(m) - d.sum()) < 1e-5, "rotated diagonal symmetric + trace")

q = axis_angle([2, 0, -1], 0.8)
d = np.array([0.7, 1.1, 2.3], dtype=f32); v = np.array([0.4, -1.2, 0.9], dtype=f32)
via_matrix = rotated_diagonal(q, d) @ v
via_rot = rotate(q, d * rotate(qconj(q), v))
check(np.allclose(via_matrix, via_rot, atol=1e-5), "rotated diagonal == rotate/scale/rotate back")

# rot_matrix mirrors rotation by the quaternion
q = axis_angle([1, 2, 3], 1.2); v = np.array([0.3, -0.7, 1.1], dtype=f32)
check(np.allclose(rot_matrix(q) @ v, rotate(q, v), atol=1e-5), "rot_matrix matches quaternion rotation")

a3 = np.array([1, 2, 3], dtype=f32); b3 = np.array([-2, 0.5, 4], dtype=f32)
check(np.allclose(skew(a3) @ b3, np.cross(a3, b3), atol=1e-5), "skew == cross")
check(np.allclose(outer(a3, b3)[1, 2], a3[1] * b3[2]), "outer entries")

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nAll mirror checks passed")
