#!/usr/bin/env bash
# Builds two headless drivers from the vendored upstream avbd-demo3d sources:
#   driver_up       the unmodified upstream solver (source of the golden trajectories)
#   driver_colored  upstream with step() replaced by the colored Gauss-Seidel variant in colored_step.cpp
# Usage: Tools/UpstreamGolden/build.sh [output_dir]     (needs g++ and python3; no OpenGL needed)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SRC="$ROOT/savant117-avbd-demo3d/avbd-demo3d-main/source"
OUT=${1:-$(mktemp -d)}
mkdir -p "$OUT/stub/GL" "$OUT/up" "$OUT/colored"
: > "$OUT/stub/GL/gl.h"   # solver.h includes GL headers the headless build does not use

FILES="maths.h solver.h scenes.h collide.cpp force.cpp joint.cpp manifold.cpp rigid.cpp spring.cpp solver.cpp"
for d in up colored; do
  for f in $FILES; do cp "$SRC/$f" "$OUT/$d/"; done
  cp "$HERE/driver.cpp" "$OUT/$d/"
done

# Colored copy: add the two solver fields and swap in the colored step().
python3 - "$OUT/colored" "$HERE/colored_step.cpp" <<'PY'
import sys
d, step = sys.argv[1], sys.argv[2]
h = open(f"{d}/solver.h").read()
anchor = "    float gamma; // Warmstarting decay parameter\n"
assert anchor in h, "solver.h layout changed: update the anchor in build.sh"
open(f"{d}/solver.h", "w").write(h.replace(anchor, anchor + "    bool colored = false;\n    int numColors = 0;\n", 1))
c = open(f"{d}/solver.cpp").read()
i = c.index("void Solver::step()")
open(f"{d}/solver.cpp", "w").write(c[:i] + open(step).read())
PY

# -ffp-contract=off keeps the float results independent of fused multiply-add availability.
FLAGS="-O2 -ffp-contract=off -std=c++17 -w -I $OUT/stub"
CPPS="collide.cpp force.cpp joint.cpp manifold.cpp rigid.cpp spring.cpp solver.cpp"
(cd "$OUT/up"      && g++ $FLAGS -I. driver.cpp $CPPS -o "$OUT/driver_up")
(cd "$OUT/colored" && g++ $FLAGS -I. -DCOLORED_BUILD driver.cpp $CPPS -o "$OUT/driver_colored")
echo "$OUT"
