// Headless fixture driver. All SDF and optimizer results come from pinned upstream C++.
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "scenes.h"

void number(float x) {
    if (!std::isfinite(x)) throw std::runtime_error("non-finite oracle result");
    std::cout << x;
}
void vector(float3 v) { std::cout << '['; number(v.x); std::cout << ','; number(v.y); std::cout << ','; number(v.z); std::cout << ']'; }
void bounds(AABB b) { std::cout << "{\"min\":"; vector(b.min); std::cout << ",\"max\":"; vector(b.max); std::cout << '}'; }
ShapePtr shape(const std::string &name, const std::string &assets) {
    for(const std::string warp:{"bend-","twist-","displace-"}) {
        if(name.rfind(warp,0)==0) return std::make_shared<WarpedBox>(warp=="bend-"?WarpedBox::Bend:warp=="twist-"?WarpedBox::Twist:WarpedBox::Displace,
            warp=="twist-"?float3{1.15f,.55f,.28f}:float3{1.15f,.28f,.55f},std::stof(name.substr(warp.size())));
    }
    if(name=="box") return std::make_shared<Box>(float3{.5f,.5f,.5f});
    if(name=="sphere") return std::make_shared<Sphere>(.5f);
    if(name=="capsule") return std::make_shared<Capsule>(.3f,.4f);
    if(name=="cylinder") return std::make_shared<Cylinder>(.4f,.4f);
    if(name=="cone") return std::make_shared<Cone>(.5f,.5f);
    if(name=="torus") return std::make_shared<Torus>(.5f,.2f);
    if(name=="spike-plate") return std::make_shared<SpikePlate>(float3{.7f,.08f,.7f},10,.045f,.12f,.26f);
    if(name=="spiky-wheel") return std::make_shared<SpikyWheel>(.35f,.48f,.08f,1.4f,16);
    if(name=="bend") return std::make_shared<WarpedBox>(WarpedBox::Bend,float3{1.15f,.28f,.55f},.95f);
    if(name=="twist") return std::make_shared<WarpedBox>(WarpedBox::Twist,float3{1.15f,.55f,.28f},1.25f);
    if(name=="displace") return std::make_shared<WarpedBox>(WarpedBox::Displace,float3{1.15f,.28f,.55f},.1f);
    if(name=="bunny" || name=="gear" || name=="cow" || name=="chair") {
        Mesh mesh; std::string error;
        if(!loadObj(assets+"/"+name+".obj",mesh,error)) throw std::runtime_error(error);
        return std::make_shared<VoxelSdf>(mesh,name,100);
    }
    throw std::runtime_error("unknown shape: "+name);
}
void samples(const std::string &name,const std::string &assets) {
    auto s=shape(name,assets);
    std::cout << "{\"shape\":\"" << name << "\",\"bounds\":"; bounds(s->bounds);
    if(auto w=dynamic_cast<WarpedBox*>(s.get())) {
        std::cout << ",\"warp\":{\"type\":"<<int(w->type)<<",\"k\":";number(w->k);
        std::cout << ",\"halfExtents\":";vector(w->halfExtents);std::cout << '}';
    }
    if(auto v=dynamic_cast<VoxelSdf*>(s.get())) {
        std::cout << ",\"grid\":{\"dimensions\":["<<v->nx<<','<<v->ny<<','<<v->nz<<"],\"cell\":"; number(v->cell);
        std::cout << ",\"origin\":"; vector(v->origin); std::cout << ",\"bytes\":"<<v->grid.size()*sizeof(float4)<<'}';
    }
    std::cout << ",\"samples\":[";
    // A fixed grid includes interior, exterior, axes, corners and exact primitive surfaces.
    bool first=true;
    for(float x:{-.75f,-.5f,-.13f,0.f,.17f,.5f,.75f})
    for(float y:{-.75f,-.5f,0.f,.5f,.75f})
    for(float z:{-.75f,-.5f,0.f,.5f,.75f}) {
        float3 p{x,y,z},g; float d=s->sdf(p,&g);
        if(!first) std::cout << ','; first=false;
        std::cout << "{\"p\":";vector(p);std::cout << ",\"phi\":";number(d);std::cout << ",\"gradient\":";vector(g);std::cout << '}';
    }
    std::cout << "]}\n";
}
void optimize() {
    std::cout << "{\"cases\":["; bool first=true;
    for(const std::string name:{"box","sphere","torus","bend"})
    for(int variant=0;variant<4;variant++) {
        auto a=shape(name,""),b=shape("box","");
        // Static bodies avoid mass-property setup: only sdf() and world transforms are used.
        float3 origin{.35f,.1f,0}; quat rotation{0,0,.258819045f,.965925826f};
        Body bodyA(a,{0,0,0},{0,0,0,1},1,true),bodyB(b,origin,rotation,1,true);
        AABB domain=intersect(bodyA.bounds,bodyB.bounds);
        float3 seed=variant==0?float3{2,-2,2}:variant==1?domain.min:variant==2?domain.max:(domain.min+domain.max)*.5f;
        float3 result=seed,step{.05f,.07f,.09f};
        std::vector<ContactPoint> manifold;
        if(variant==3) { ContactPoint p; p.x={.1f,.1f,.1f};manifold.push_back(p); }
        int queries=0,limit=variant==2?0:50;float epsilon=variant==1?0.f:.1f;
        Objective objective{&bodyA,&bodyB,&manifold,epsilon,.01f,&queries};
        int iterations=nelderMead(objective,result,domain,step,.01f,limit);
        const int optimizerQueries=queries; float value=objective(result);
        if(!first)std::cout<<',';first=false;
        std::cout<<"{\"shapeA\":\""<<name<<"\",\"shapeB\":\"box\",\"originB\":";vector(origin);
        std::cout<<",\"rotationB\":[0,0,0.258819045,0.965925826],\"domain\":";bounds(domain);
        std::cout<<",\"seed\":";vector(seed);std::cout<<",\"initialStep\":";vector(step);
        std::cout<<",\"epsilon\":";number(epsilon);std::cout<<",\"alpha\":0.01,\"tolerance\":0.01,\"maxIterations\":"<<limit;
        std::cout<<",\"repulsionPoints\":[";if(!manifold.empty())vector(manifold[0].x);std::cout<<"],\"x\":";vector(result);
        std::cout<<",\"objective\":";number(value);std::cout<<",\"iterations\":"<<iterations<<",\"queries\":"<<optimizerQueries<<'}';
    }
    std::cout<<"]}\n";
}
void quaternion(quat q) { std::cout<<'[';number(q.x);std::cout<<',';number(q.y);std::cout<<',';number(q.z);std::cout<<',';number(q.w);std::cout<<']'; }
void matrix(float3x3 m) {
    std::cout<<'[';
    for(int i=0;i<3;i++) { if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<3;j++){if(j)std::cout<<',';number(m[i][j]);}std::cout<<']'; }
    std::cout<<']';
}
void points(const std::vector<ContactPoint> &ps) {
    std::cout<<'[';
    for(size_t i=0;i<ps.size();i++) {
        if(i)std::cout<<',';const auto &p=ps[i];
        std::cout<<"{\"x\":";vector(p.x);std::cout<<",\"normal\":";vector(p.normal);
        std::cout<<",\"depth\":";number(p.depth);std::cout<<",\"separation\":";number(p.separation);
        std::cout<<",\"softmax\":";number(p.softmax);std::cout<<",\"rank\":";number(p.rank);
        std::cout<<",\"active\":"<<(p.active?"true":"false")<<",\"cached\":"<<(p.cached?"true":"false");
        std::cout<<",\"normalImpulse\":";number(p.normalImpulse);std::cout<<",\"frictionImpulse\":";vector(p.frictionImpulse);
        std::cout<<",\"ellipsoid\":";matrix(p.ellipsoid);std::cout<<'}';
    }
    std::cout<<']';
}
void parameters(const CollisionParams &p) {
    std::cout<<"{\"optimizer\":"<<p.optimizer<<",\"cacheSize\":"<<p.cacheSize<<",\"randomSamples\":"<<p.randomSamples;
    std::cout<<",\"warmStart\":"<<(p.warmStart?"true":"false")<<",\"curvatureSampling\":"<<(p.curvatureSampling?"true":"false")
             <<",\"skipConvexSamples\":"<<(p.skipConvexSamples?"true":"false");
    std::cout<<",\"epsilon\":";number(p.epsilon);std::cout<<",\"alpha\":";number(p.alpha);
    std::cout<<",\"replacementThreshold\":";number(p.replacementThreshold);std::cout<<",\"tolerance\":";number(p.tolerance);
    std::cout<<",\"maxIterations\":"<<p.maxIterations<<",\"stepSize\":";number(p.stepSize);
    std::cout<<",\"margin\":";number(p.margin);std::cout<<",\"allowedPenetration\":";number(p.allowedPenetration);std::cout<<'}';
}
void shapeDescription(const Shape &s) {
    std::cout<<"{\"bounds\":";bounds(s.bounds);std::cout<<",\"centerOfMass\":";vector(s.centerOfMass);
    std::cout<<",\"convex\":"<<(s.convex?"true":"false");
    if(auto p=dynamic_cast<const Box*>(&s)) {std::cout<<",\"kind\":\"box\",\"halfExtents\":";vector(p->halfExtents);}
    else if(auto p=dynamic_cast<const Sphere*>(&s)) {std::cout<<",\"kind\":\"sphere\",\"radius\":";number(p->radius);}
    else if(auto p=dynamic_cast<const Capsule*>(&s)) {std::cout<<",\"kind\":\"capsule\",\"radius\":";number(p->radius);std::cout<<",\"halfLength\":";number(p->halfLength);}
    else if(auto p=dynamic_cast<const Cylinder*>(&s)) {std::cout<<",\"kind\":\"cylinder\",\"radius\":";number(p->radius);std::cout<<",\"halfHeight\":";number(p->halfHeight);}
    else if(auto p=dynamic_cast<const Cone*>(&s)) {std::cout<<",\"kind\":\"cone\",\"radius\":";number(p->radius);std::cout<<",\"halfHeight\":";number(p->halfHeight);}
    else if(auto p=dynamic_cast<const Torus*>(&s)) {std::cout<<",\"kind\":\"torus\",\"majorRadius\":";number(p->majorRadius);std::cout<<",\"minorRadius\":";number(p->minorRadius);}
    else if(auto p=dynamic_cast<const WarpedBox*>(&s)) {std::cout<<",\"kind\":\"warp\",\"type\":"<<p->type<<",\"k\":";number(p->k);std::cout<<",\"halfExtents\":";vector(p->halfExtents);}
    else if(auto p=dynamic_cast<const VoxelSdf*>(&s)) {std::cout<<",\"kind\":\"voxel\",\"asset\":\""<<p->name<<"\",\"resolution\":100";}
    else if(auto p=dynamic_cast<const SpikePlate*>(&s)) {
        std::cout<<",\"kind\":\"spike-plate\",\"halfExtents\":";vector(p->halfExtents);std::cout<<",\"count\":"<<p->count;
        std::cout<<",\"spikeRadius\":";number(p->spikeRadius);std::cout<<",\"minHeight\":";number(p->minHeight);std::cout<<",\"maxHeight\":";number(p->maxHeight);
    } else if(auto p=dynamic_cast<const SpikyWheel*>(&s)) {
        std::cout<<",\"kind\":\"spiky-wheel\",\"spikes\":"<<p->spikes<<",\"hubRadius\":";number(p->hubRadius);
        std::cout<<",\"hubHalfWidth\":";number(p->hubHalfWidth);std::cout<<",\"spikeRadius\":";number(p->spikeRadius);std::cout<<",\"spikeHalfHeight\":";number(p->spikeHalfHeight);
    } else throw std::runtime_error("unrecorded shape type");
    std::cout<<'}';
}
void body(const Body &b) {
    std::cout<<"{\"origin\":";vector(b.toWorld({0,0,0}));std::cout<<",\"position\":";vector(b.position);
    std::cout<<",\"orientation\":";quaternion(b.orientation);std::cout<<",\"scale\":";number(b.scale);
    std::cout<<",\"static\":"<<(b.isStatic?"true":"false")<<",\"velocity\":";vector(b.velocity);
    std::cout<<",\"angularVelocity\":";vector(b.angularVelocity);std::cout<<",\"bounds\":";bounds(b.bounds);
    std::cout<<",\"shape\":";shapeDescription(*b.shape);std::cout<<",\"friction\":";number(b.friction);
    std::cout<<",\"mass\":";number(b.mass);std::cout<<",\"inertia\":";matrix(b.inertia);std::cout<<'}';
}
void statistics(const CollisionStats &s) {
    std::cout<<"{\"pairs\":"<<s.pairs<<",\"contacts\":"<<s.contacts<<",\"iterations\":"<<s.iterations<<",\"queries\":"<<s.queries<<'}';
}
void seeds(const std::string &name,const std::string &assets) {
    auto s=shape(name,assets);s->finalize();float L=maxComponent(size(s->bounds));
    std::cout<<"{\"shape\":\""<<name<<"\",\"bounds\":";bounds(s->bounds);
    std::cout<<",\"convex\":"<<(s->convex?"true":"false")<<",\"volume\":";number(s->volume);
    std::cout<<",\"centerOfMass\":";vector(s->centerOfMass);std::cout<<",\"inertia\":";matrix(s->inertia);
    std::cout<<",\"kappa\":";number(4.f/L);std::cout<<",\"delta\":";number(.001f*L);
    std::cout<<",\"spacing\":";number(.08f*L);std::cout<<",\"seeds\":[";
    for(size_t i=0;i<s->seeds.size();i++) {
        if(i)std::cout<<',';auto p=s->seeds[i];std::cout<<"{\"p\":";vector(p);
        std::cout<<",\"phi\":";number(s->sdf(p));std::cout<<",\"laplacian\":";number(s->laplacian(p,.001f*L));std::cout<<'}';
    }
    std::cout<<"]}\n";
}
// Recover the exact uint32 draws consumed by one update from PCG's public state.
// This observes the unmodified algorithm: no hooks or replacements in Random::next().
void randomDraws(Random before,const Random &after) {
    std::cout<<"{\"before\":\""<<before.state<<"\",\"after\":\""<<after.state<<"\",\"draws\":[";
    size_t n=0;
    while(before.state!=after.state) {
        if(n==1000000)throw std::runtime_error("RNG replay exceeded bound");
        if(n++)std::cout<<',';std::cout<<before.next();
    }
    std::cout<<"]}";
}
// optimizers=true replays Ellipsoid/gradient-descent cases into a separate fixture.
void manifolds(const std::string &assets,bool optimizers=false) {
    std::cout<<"{\"cases\":[";bool first=true;
    const std::vector<std::string> cases=optimizers?std::vector<std::string>{"ellipsoid-warm-box","ellipsoid-torus-seeds","sgd-warm-box","sgd-bend-seeds"}:
        std::vector<std::string>{"cold-box","warm-box","static-a-advection","static-b-advection","dynamic-advection",
        "convex-skip","convex-no-skip","duplicate-cache","warm-disabled","no-cache","margin-gap","separated-spheres",
        "scaled-rotated","replacement-zero","replacement-high","torus-seeds","bend-seeds","voxel-seeds"};
    for(const auto &name:cases) {
        const bool torus=name.find("torus-seeds")!=std::string::npos,bend=name.find("bend-seeds")!=std::string::npos;
        std::string an=torus?"torus":bend?"bend":name=="voxel-seeds"?"bunny":name=="separated-spheres"?"sphere":"box";
        std::string bn=name=="separated-spheres"?"sphere":"box";
        auto a=shape(an,assets),b=shape(bn,assets);a->finalize();b->finalize();
        float3 originB=name=="margin-gap"?float3{0,1.01f,0}:name=="separated-spheres"?float3{.85f,.85f,0}:torus?float3{.35f,.1f,0}:bend?float3{0,.3f,0}:float3{0,.85f,0};
        quat rot=name=="scaled-rotated"?axisAngle(normalize(float3{1,2,3}),.4f):quat{0,0,0,1};
        Body A(a,{0,0,0},{0,0,0,1},name=="scaled-rotated"?1.3f:1.f,name=="static-a-advection");
        Body B(b,originB,rot,name=="scaled-rotated"?.7f:1.f,name=="static-b-advection");
        CollisionParams p;float dt=1.f/60.f;Random rng;
        bool advect=name.find("advection")!=std::string::npos;
        if(advect) {A.velocity={.2f,.03f,-.1f};A.angularVelocity={.1f,.2f,.3f};B.velocity={-.1f,-.02f,.3f};B.angularVelocity={-.2f,.1f,-.15f};}
        if(name.find("ellipsoid-")==0)p.optimizer=OPTIMIZER_ELLIPSOID;
        if(name.find("sgd-")==0)p.optimizer=OPTIMIZER_GRADIENT_DESCENT;
        if(name=="convex-no-skip")p.skipConvexSamples=false;
        if(name=="warm-disabled")p.warmStart=false;
        if(name=="no-cache")p.cacheSize=0;
        if(name.find("replacement-")==0)p.skipConvexSamples=false;
        if(name=="duplicate-cache") {p.alpha=0;p.maxIterations=0;}
        if(name=="replacement-zero")p.replacementThreshold=0;
        if(name=="replacement-high")p.replacementThreshold=1;
        // Flat-face cached inputs make skip and dedup branches explicit and replayable.
        Manifold m;m.bodyA=&A;m.bodyB=&B;
        bool supplied=advect || name=="convex-skip" || name=="convex-no-skip" || name=="duplicate-cache" || name=="warm-disabled" || name.find("replacement-")==0;
        if(supplied)for(int i=0;i<4;i++) {
            ContactPoint cp;cp.x={i&1?.3f:-.3f,.425f,i&2?.3f:-.3f};
            if(name=="duplicate-cache")cp.x={0,.425f,0};
            cp.normal={0,1,0};cp.active=true;cp.normalImpulse=float(i+1);cp.frictionImpulse={.01f*float(i+1),0,0};m.points.push_back(cp);
        }
        int steps=name.find("warm-box")!=std::string::npos?4:1;
        for(int step=0;step<steps;step++) {
            if(!first)std::cout<<',';first=false;
            std::cout<<"{\"name\":\""<<name<<"\",\"step\":"<<step<<",\"shapeA\":\""<<an<<"\",\"shapeB\":\""<<bn<<"\",\"bodyA\":";body(A);
            std::cout<<",\"bodyB\":";body(B);std::cout<<",\"params\":";parameters(p);std::cout<<",\"dt\":";number(dt);
            std::cout<<",\"initialCache\":";points(m.points);auto before=rng;CollisionStats stats;
            m.update(p,dt,rng,stats);
            std::cout<<",\"domain\":";bounds(m.volume);std::cout<<",\"rng\":";randomDraws(before,rng);
            std::cout<<",\"points\":";points(m.points);std::cout<<",\"stats\":";statistics(stats);std::cout<<'}';
        }
    }
    std::cout<<"]}\n";
}
void sceneMetrics(int index,int frames,const std::string &warpK) {
    if(index<0 || index>=sceneCount || frames<1 || frames>3600)throw std::runtime_error("invalid scene/frames");
    Solver solver;Camera camera;scenes[index].build(solver,camera);
    // Table 5 strengths on the released falling-box scene, not a paper-scene reconstruction.
    if(!warpK.empty()) {
        if(index<3 || index>5)throw std::runtime_error("warp override requires a warped-box scene");
        auto old=dynamic_cast<WarpedBox*>(solver.bodies[1]->shape.get());
        auto replacement=std::make_shared<WarpedBox>(old->type,old->halfExtents,std::stof(warpK));replacement->finalize();
        for(auto &b:solver.bodies)if(!b->isStatic) {
            auto next=std::make_unique<Body>(replacement,b->toWorld({0,0,0}),b->orientation,b->scale);
            next->velocity=b->velocity;next->angularVelocity=b->angularVelocity;next->friction=b->friction;b=std::move(next);
        }
    }
    std::cout<<"{\"scene\":\""<<scenes[index].name<<"\",\"sceneIndex\":"<<index<<",\"frames\":"<<frames;
    std::cout<<",\"warpOverride\":\""<<warpK<<"\"";
    std::cout<<",\"solver\":{\"type\":\"upstream sequential impulses\",\"dt\":";number(solver.dt);
    std::cout<<",\"substeps\":"<<solver.substeps<<",\"iterations\":"<<solver.iterations<<",\"gravity\":";vector(solver.gravity);
    std::cout<<",\"baumgarte\":";number(solver.baumgarte);std::cout<<",\"maxCorrection\":";number(solver.maxCorrection);
    std::cout<<"},\"params\":";parameters(solver.params);std::cout<<",\"initialBodies\":[";
    for(size_t i=0;i<solver.bodies.size();i++){if(i)std::cout<<',';body(*solver.bodies[i]);}
    std::cout<<"],\"samples\":[";
    for(int frame=1;frame<=frames;frame++) {
        solver.step();float energy=0,penetration=0,minY=std::numeric_limits<float>::max();
        for(auto &b:solver.bodies)if(!b->isStatic) {
            float3 w=rotate(conjugate(b->orientation),b->angularVelocity);
            energy+=.5f*(b->mass*lengthSq(b->velocity)+dot(w,b->inertia*w));minY=min(minY,b->position.y);
            for(float x:{b->position.x,b->position.y,b->position.z,b->orientation.x,b->orientation.y,b->orientation.z,b->orientation.w})
                if(!std::isfinite(x))throw std::runtime_error("non-finite scene pose");
        }
        for(auto &entry:solver.manifolds)for(auto &p:entry.second.points)if(p.active)penetration=max(penetration,-p.separation);
        if(frame>1)std::cout<<',';std::cout<<"{\"frame\":"<<frame<<",\"stats\":";statistics(solver.stats);
        std::cout<<",\"kineticEnergy\":";number(energy);std::cout<<",\"maxContactPenetration\":";number(penetration);
        std::cout<<",\"minCenterY\":";number(minY);
        // Independent geometry probe: unlike contact depth, this can see missed contacts.
        // Evaluate every mesh vertex against other overlapping bodies every 30 frames.
        if(frame%30==0) {
            float vertexPenetration=0;int probes=0;
            for(auto &b:solver.bodies)b->updateBounds();
            for(auto &a:solver.bodies)for(auto &b:solver.bodies) {
                if(a.get()==b.get() || (a->isStatic && b->isStatic) || !overlaps(a->bounds,b->bounds))continue;
                for(auto p:a->shape->mesh.positions) {float d=b->sdf(a->toWorld(p));if(!std::isfinite(d))throw std::runtime_error("non-finite vertex probe");vertexPenetration=max(vertexPenetration,-d);probes++;}
            }
            std::cout<<",\"maxVertexPenetration\":";number(vertexPenetration);std::cout<<",\"vertexQueries\":"<<probes;
        }
        std::cout<<'}';
    }
    std::cout<<"]}\n";
}

// Ellipsoid and gradient-descent optimizers plus the analytic objective gradient (Sec. 3.3, Table 5).
// Same bodies, domains and seeds as optimize(); each case also replays an ellipsoid warm start.
void optimizers() {
    std::cout << "{\"cases\":["; bool first=true;
    for(const std::string name:{"box","sphere","torus","bend"})
    for(int variant=0;variant<4;variant++) {
        auto a=shape(name,""),b=shape("box","");
        float3 origin{.35f,.1f,0}; quat rotation{0,0,.258819045f,.965925826f};
        Body bodyA(a,{0,0,0},{0,0,0,1},1,true),bodyB(b,origin,rotation,1,true);
        AABB domain=intersect(bodyA.bounds,bodyB.bounds);
        float3 seed=variant==0?float3{2,-2,2}:variant==1?domain.min:variant==2?domain.max:(domain.min+domain.max)*.5f;
        std::vector<ContactPoint> manifold;
        if(variant==3) { ContactPoint p; p.x={.1f,.1f,.1f};manifold.push_back(p); }
        int limit=variant==2?0:50;float epsilon=variant==1?0.f:.1f;
        int queries=0;
        Objective objective{&bodyA,&bodyB,&manifold,epsilon,.01f,&queries};
        float3 gradient;float value=objective(seed,&gradient);
        // collide.cpp: an ellipsoid centred at x enclosing V, warm started 0.75/0.25 when cached.
        float3 extent=size(domain)*0.5f+abs(seed-center(domain));
        float3x3 E0=diagonal(extent*extent*3.0f),E=E0;float3 ellipsoid=seed;
        queries=0;int ellipsoidIterations=ellipsoidMethod(objective,ellipsoid,domain,E,.01f,limit);int ellipsoidQueries=queries;
        float3 warmSeed=ellipsoid;float3 warmExtent=size(domain)*0.5f+abs(warmSeed-center(domain));
        float3x3 warmStart=E*0.75f+diagonal(warmExtent*warmExtent*3.0f)*0.25f,warmE=warmStart;float3 warm=warmSeed;
        queries=0;int warmIterations=ellipsoidMethod(objective,warm,domain,warmE,.01f,limit);int warmQueries=queries;
        Random rng;float3 descent=seed;
        queries=0;int descentIterations=gradientDescent(objective,descent,domain,.1f,.01f,limit,rng);int descentQueries=queries;
        if(!first)std::cout<<',';first=false;
        std::cout<<"{\"shapeA\":\""<<name<<"\",\"shapeB\":\"box\",\"originB\":";vector(origin);
        std::cout<<",\"rotationB\":[0,0,0.258819045,0.965925826],\"domain\":";bounds(domain);
        std::cout<<",\"seed\":";vector(seed);std::cout<<",\"epsilon\":";number(epsilon);
        std::cout<<",\"alpha\":0.01,\"tolerance\":0.01,\"maxIterations\":"<<limit<<",\"stepSize\":0.1";
        std::cout<<",\"repulsionPoints\":[";if(!manifold.empty())vector(manifold[0].x);std::cout<<"]";
        std::cout<<",\"seedValue\":";number(value);std::cout<<",\"seedGradient\":";vector(gradient);
        std::cout<<",\"ellipsoid\":{\"initial\":";matrix(E0);std::cout<<",\"x\":";vector(ellipsoid);std::cout<<",\"final\":";matrix(E);
        std::cout<<",\"iterations\":"<<ellipsoidIterations<<",\"queries\":"<<ellipsoidQueries<<'}';
        std::cout<<",\"warmEllipsoid\":{\"seed\":";vector(warmSeed);std::cout<<",\"initial\":";matrix(warmStart);std::cout<<",\"x\":";vector(warm);std::cout<<",\"final\":";matrix(warmE);
        std::cout<<",\"iterations\":"<<warmIterations<<",\"queries\":"<<warmQueries<<'}';
        std::cout<<",\"gradientDescent\":{\"x\":";vector(descent);std::cout<<",\"iterations\":"<<descentIterations<<",\"queries\":"<<descentQueries<<"}}";
    }
    std::cout<<"]}\n";
}
int main(int argc,char **argv) {
    try {
        std::cout<<std::setprecision(std::numeric_limits<float>::max_digits10);
        if(argc==2 && std::string(argv[1])=="optimize") optimize();
        else if(argc==2 && std::string(argv[1])=="optimizers") optimizers();
        else if(argc==4 && std::string(argv[1])=="sample") samples(argv[2],argv[3]);
        else if(argc==4 && std::string(argv[1])=="seeds") seeds(argv[2],argv[3]);
        else if(argc==3 && std::string(argv[1])=="manifolds") manifolds(argv[2]);
        else if(argc==3 && std::string(argv[1])=="manifolds-optimizers") manifolds(argv[2],true);
        else if((argc==4 || argc==5) && std::string(argv[1])=="scene") sceneMetrics(std::stoi(argv[2]),std::stoi(argv[3]),argc==5?argv[4]:"");
        else throw std::runtime_error("usage: oracle optimize | optimizers | sample/seeds SHAPE ASSETS | manifolds ASSETS | scene INDEX FRAMES");
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
