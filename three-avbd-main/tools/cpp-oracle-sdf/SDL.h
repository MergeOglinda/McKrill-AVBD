// Headless adapters for scenes.cpp's asset lookup and error reporting only.
// No SDL library, window, renderer, or GPU device is created.
#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
inline char *SDL_GetBasePath() {
    const char *root = std::getenv("SDF_ORACLE_ROOT");
    if (!root) throw std::runtime_error("SDF_ORACLE_ROOT is required");
    return ::strdup(root);
}
inline void SDL_free(void *p) { std::free(p); }
// Upstream otherwise substitutes a box after an asset failure. Never accept that as a golden.
inline void SDL_Log(const char *, const char *error) { throw std::runtime_error(error); }
