// Export upstream normalized surface triangles for CPU seed parity; no engine-generated goldens.
#include <fstream>
#include <cstring>
#include <iostream>
#include "shapes.h"
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    Mesh mesh; std::string error;
    if (!loadObj(std::string(argv[1]) + "/" + argv[2] + ".obj", mesh, error)) { std::cerr << error; return 1; }
    VoxelSdf field(mesh, argv[2], 100);
    static_assert(sizeof(float4) == 16);
    // Emit explicit little-endian float32 bytes, regardless of host endianness.
    std::ofstream out(argv[3], std::ios::binary);
    for (int t = 0; t < field.mesh.triangleCount(); t++) for (int c = 0; c < 3; c++) for (float f : {field.mesh.vertex(t,c).x, field.mesh.vertex(t,c).y, field.mesh.vertex(t,c).z}) {
        uint32_t bits; static_assert(sizeof(bits) == sizeof(f));
        std::memcpy(&bits, &f, 4);
        for (int i = 0; i < 4; i++) out.put(char((bits >> (8 * i)) & 255));
    }
    return out.good() ? 0 : 1;
}
