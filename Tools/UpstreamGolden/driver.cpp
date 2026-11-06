// Scratch driver. Built twice: against upstream sources (golden) and against the colored prototype.
#include "scenes.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>

static std::vector<Rigid *> creationOrder(Solver *s)
{
    std::vector<Rigid *> v;
    for (Rigid *b = s->bodies; b; b = b->next)
        v.push_back(b);
    std::reverse(v.begin(), v.end());
    return v;
}


static float jointResidual(Solver *s)
{
    float worst = 0.0f;
    for (Force *f = s->forces; f; f = f->next)
        if (Joint *j = dynamic_cast<Joint *>(f))
        {
            float3 a = j->bodyA ? transform(j->bodyA->positionLin, j->bodyA->positionAng, j->rA) : j->rA;
            float3 b = transform(j->bodyB->positionLin, j->bodyB->positionAng, j->rB);
            worst = std::max(worst, length(a - b));
        }
    return worst;
}

static void build(const char *name, Solver *s)
{
    if (!strcmp(name, "ground")) sceneGround(s);
    else if (!strcmp(name, "stack")) sceneStack(s);
    else if (!strcmp(name, "rope")) sceneRope(s);
    else if (!strcmp(name, "spring")) sceneSpring(s);
    else if (!strcmp(name, "pyramid")) scenePyramid(s);
    else if (!strcmp(name, "stackratio")) sceneStackRatio(s);
    else { fprintf(stderr, "unknown scene %s\n", name); exit(1); }
}

static void dump(const char *name, int step, Solver *s)
{
    auto v = creationOrder(s);
    printf("S %s %d %zu\n", name, step, v.size());
    for (Rigid *b : v)
        printf("B %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
               b->positionLin.x, b->positionLin.y, b->positionLin.z,
               b->positionAng.x, b->positionAng.y, b->positionAng.z, b->positionAng.w,
               b->velocityLin.x, b->velocityLin.y, b->velocityLin.z,
               b->velocityAng.x, b->velocityAng.y, b->velocityAng.z);
}

int main(int argc, char **argv)
{
    // usage: driver golden <scene> <steps...>
    //        driver jitter <scene> <eps> <steps...>   (perturb dynamic bodies' x by eps)
    //        driver compare <scene> <steps>           (colored vs sequential; needs colored build)
    const char *mode = argv[1];
    const char *scene = argv[2];

    if (!strcmp(mode, "golden") || !strcmp(mode, "jitter"))
    {
        float eps = 0.0f;
        int first = 3;
        if (!strcmp(mode, "jitter")) { eps = (float)atof(argv[3]); first = 4; }
        Solver s;
        build(scene, &s);
        if (eps != 0.0f)
            for (Rigid *b : creationOrder(&s))
                if (b->mass > 0) b->positionLin.x += eps;
        int last = 0;
        for (int i = first; i < argc; i++) last = std::max(last, atoi(argv[i]));
        for (int step = 1; step <= last; step++)
        {
            s.step();
            for (int i = first; i < argc; i++)
                if (atoi(argv[i]) == step) dump(scene, step, &s);
        }
        return 0;
    }

#ifdef COLORED_BUILD
    if (!strcmp(mode, "compare"))
    {
        int steps = atoi(argv[3]);
        Solver a, b;
        build(scene, &a);
        build(scene, &b);
        b.colored = true;
        float worstEver = 0.0f;
        for (int step = 1; step <= steps; step++)
        {
            a.step();
            b.step();
            auto va = creationOrder(&a), vb = creationOrder(&b);
            float worst = 0.0f, vmax = 0.0f;
            for (size_t i = 0; i < va.size(); i++)
            {
                float3 d = va[i]->positionLin - vb[i]->positionLin;
                worst = std::max(worst, length(d));
                vmax = std::max(vmax, length(vb[i]->velocityLin));
            }
            worstEver = std::max(worstEver, worst);
            if (step == 1 || step % 30 == 0 || step == steps)
                printf("%s step %d  max|dpos seq-vs-colored| = %.5f  colors=%d  max|v colored| = %.4f  jointRes seq=%.5f col=%.5f\n", scene, step, worst, b.numColors, vmax, jointResidual(&a), jointResidual(&b));
        }
        // final resting summary for colored run
        auto vb = creationOrder(&b);
        printf("%s colored final z:", scene);
        for (size_t i = 0; i < vb.size() && i < 12; i++) printf(" %.4f", vb[i]->positionLin.z);
        printf("\n%s sequential final z:", scene);
        auto va = creationOrder(&a);
        for (size_t i = 0; i < va.size() && i < 12; i++) printf(" %.4f", va[i]->positionLin.z);
        printf("\n");
    }
#endif
    return 0;
}
