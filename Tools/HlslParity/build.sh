#!/usr/bin/env bash
# Builds collide_parity and contact_parity: upstream avbd-demo3d's unmodified solver, plus the package's contact HLSL
# (Shaders/Include/AvbdBoxCollision.hlsl, AvbdContacts.hlsl, AvbdContactWarmStart.hlsl, AvbdContactSolve.hlsl) compiled as C++ through hlsl_shim.h.
# collide_parity compares the narrowphase with upstream collide() on demo scenes and random box pairs; contact_parity compares the warm start,
# primal terms and dual update with upstream's Manifold. See README.md.
# Usage: Tools/HlslParity/build.sh [output_dir]     (needs g++ and perl; no GPU or OpenGL needed)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
SRC="$ROOT/savant117-avbd-demo3d/avbd-demo3d-main/source"
INC=Packages/com.mckrill.avbd/Shaders/Include
OUT=${1:-$(mktemp -d)}
mkdir -p "$OUT/stub/GL" "$OUT/work" "$OUT/hlsl/$INC"
: > "$OUT/stub/GL/gl.h"   # solver.h includes GL headers the headless build does not use

for f in maths.h solver.h scenes.h collide.cpp force.cpp joint.cpp manifold.cpp rigid.cpp spring.cpp solver.cpp; do
    cp "$SRC/$f" "$OUT/work/"
done
cp "$HERE/hlsl_shim.h" "$HERE/parity_random.h" "$HERE/collide_parity.cpp" "$HERE/contact_parity.cpp" "$OUT/work/"

# HLSL float literals are float; in C++ they would be double and change the rounding. Give them an f suffix.
for f in AvbdQuaternion AvbdBoxCollision AvbdContacts AvbdCommon AvbdContactSolve AvbdContactWarmStart; do
    perl -pe 's/(?<![\w.])(\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+|\.\d+)(?![\w.])/$1f/g' "$ROOT/Packages/com.mckrill.avbd/Shaders/Include/$f.hlsl" > "$OUT/hlsl/$INC/$f.hlsl"
done

# -ffp-contract=off keeps the float results independent of fused multiply-add availability.
FLAGS="-O2 -ffp-contract=off -std=c++17 -w -I $OUT/stub -I $OUT/hlsl"
UPSTREAM="collide.cpp force.cpp joint.cpp manifold.cpp rigid.cpp spring.cpp solver.cpp"
(cd "$OUT/work" && g++ $FLAGS -I. collide_parity.cpp $UPSTREAM -o "$OUT/collide_parity" && g++ $FLAGS -I. contact_parity.cpp $UPSTREAM -o "$OUT/contact_parity")
echo "$OUT"
