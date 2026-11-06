#include <vector>
#include <map>
#include <algorithm>

// Colored Gauss-Seidel variant of upstream Solver::step(): identical except that the primal sweep visits
// bodies color by color (greedy first-fit, newest body first). Compiled in place of the upstream step()
// by build.sh; it is what the C# GaussSeidelMode.Colored mirrors.

void Solver::step()
{
    for (Rigid *bodyA = bodies; bodyA != 0; bodyA = bodyA->next)
    {
        for (Rigid *bodyB = bodyA->next; bodyB != 0; bodyB = bodyB->next)
        {
            float3 dp = bodyA->positionLin - bodyB->positionLin;
            float r = bodyA->radius + bodyB->radius;
            if (dot(dp, dp) <= r * r && !bodyA->constrainedTo(bodyB))
                new Manifold(this, bodyA, bodyB);
        }
    }

    for (Force *force = forces; force != 0;)
    {
        if (!force->initialize())
        {
            Force *next = force->next;
            delete force;
            force = next;
        }
        else
            force = force->next;
    }

    for (Rigid *body = bodies; body != 0; body = body->next)
    {
        body->inertialLin = body->positionLin + body->velocityLin * dt;
        if (body->mass > 0)
            body->inertialLin += float3{0, 0, gravity} * (dt * dt);
        body->inertialAng = body->positionAng + body->velocityAng * dt;

        float3 accel = (body->velocityLin - body->prevVelocityLin) / dt;
        float accelExt = accel.z * sign(gravity);
        float accelWeight = clamp(accelExt / abs(gravity), 0.0f, 1.0f);
        if (!isfinite(accelWeight))
            accelWeight = 0.0f;

        body->initialLin = body->positionLin;
        body->initialAng = body->positionAng;
        if (body->mass > 0)
        {
            body->positionLin = body->positionLin + body->velocityLin * dt + float3{0, 0, gravity} * (accelWeight * dt * dt);
            body->positionAng = body->positionAng + body->velocityAng * dt;
        }
    }

    // Greedy first-fit coloring over dynamic bodies, in list order.
    std::vector<std::vector<Rigid *>> groups;
    if (colored)
    {
        std::map<Rigid *, int> color;
        for (Rigid *body = bodies; body != 0; body = body->next)
        {
            if (body->mass <= 0)
                continue;
            std::vector<bool> used;
            for (Force *f = body->forces; f != 0; f = (f->bodyA == body) ? f->nextA : f->nextB)
            {
                if (dynamic_cast<IgnoreCollision *>(f))
                    continue;
                Rigid *other = f->bodyA == body ? f->bodyB : f->bodyA;
                if (!other || other->mass <= 0)
                    continue;
                auto it = color.find(other);
                if (it != color.end())
                {
                    if ((int)used.size() <= it->second)
                        used.resize(it->second + 1, false);
                    used[it->second] = true;
                }
            }
            int c = 0;
            while (c < (int)used.size() && used[c])
                c++;
            color[body] = c;
            if ((int)groups.size() <= c)
                groups.resize(c + 1);
            groups[c].push_back(body);
        }
        numColors = (int)groups.size();
    }

    auto primalUpdate = [&](Rigid *body)
    {
        if (body->mass <= 0)
            return;

        float3x3 MLin = diagonal(body->mass, body->mass, body->mass);
        float3x3 MAng = diagonal(body->moment.x, body->moment.y, body->moment.z);

        float3x3 lhsLin = MLin / (dt * dt);
        float3x3 lhsAng = MAng / (dt * dt);
        float3x3 lhsCross = float3x3{0, 0, 0, 0, 0, 0, 0, 0, 0};

        float3 rhsLin = MLin / (dt * dt) * (body->positionLin - body->inertialLin);
        float3 rhsAng = MAng / (dt * dt) * (body->positionAng - body->inertialAng);

        for (Force *force = body->forces; force != 0; force = (force->bodyA == body) ? force->nextA : force->nextB)
            force->updatePrimal(body, alpha, lhsLin, lhsAng, lhsCross, rhsLin, rhsAng);

        float3 dxLin, dxAng;
        solve(lhsLin, lhsAng, lhsCross, -rhsLin, -rhsAng, dxLin, dxAng);
        body->positionLin = body->positionLin + dxLin;
        body->positionAng = body->positionAng + dxAng;
    };

    for (int it = 0; it < iterations; it++)
    {
        if (!colored)
        {
            for (Rigid *body = bodies; body != 0; body = body->next)
                primalUpdate(body);
        }
        else
        {
            for (auto &group : groups)
                for (Rigid *body : group)
                    primalUpdate(body);
        }

        for (Force *force = forces; force != 0; force = force->next)
            force->updateDual(alpha);
    }

    for (Rigid *body = bodies; body != 0; body = body->next)
    {
        body->prevVelocityLin = body->velocityLin;
        if (body->mass > 0)
        {
            body->velocityLin = (body->positionLin - body->initialLin) / dt;
            body->velocityAng = (body->positionAng - body->initialAng) / dt;
        }
    }
}
