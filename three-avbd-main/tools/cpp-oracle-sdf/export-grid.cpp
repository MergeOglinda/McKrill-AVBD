// Export upstream float4 grids for CPU interpolation parity; no engine-generated goldens.
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
    for (auto v : field.grid) for (float f : {v.x, v.y, v.z, v.w}) {
        uint32_t bits; static_assert(sizeof(bits) == sizeof(f));
        std::memcpy(&bits, &f, 4);
        for (int i = 0; i < 4; i++) out.put(char((bits >> (8 * i)) & 255));
    }
    return out.good() ? 0 : 1;
}
