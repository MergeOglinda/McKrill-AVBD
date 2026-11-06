using Unity.Mathematics;

namespace McKrill.Avbd.Reference
{
    /// <summary>
    /// Box-box narrow phase, a port of upstream collide.cpp: separating-axis test over 15 axes, then
    /// either face clipping (up to 8 contacts) or a single edge-edge contact. Produces the contact
    /// basis and contacts with the same feature keys as upstream so warm starting matches.
    /// Written for fidelity, not speed: it allocates small scratch arrays per call.
    /// </summary>
    public static class BoxCollision
    {
        public const int MaxContacts = 8;

        const int MaxPolyVerts = 16;
        const float SatAxisEpsilon = 1.0e-6f;
        const float PlaneEpsilon = 1.0e-5f;
        const float ContactMergeDistSq = 1.0e-6f;

        // Values are packed into the top byte of the feature key, so they must not change.
        enum AxisType
        {
            FaceA = 0,
            FaceB = 1,
            Edge = 2,
        }

        struct Obb
        {
            public float3 center;
            public quaternion rotation;
            public float3 half;
            public float3 axis0;
            public float3 axis1;
            public float3 axis2;

            public float3 Axis(int i) => i == 0 ? axis0 : (i == 1 ? axis1 : axis2);
        }

        struct SatAxis
        {
            public AxisType type;
            public int indexA;
            public int indexB;
            public float separation;
            public float3 normalAB;
            public bool valid;
        }

        struct FaceFrame
        {
            public int axisIndex;
            public float3 normal;
            public float3 center;
            public float3 u;
            public float3 v;
            public float extentU;
            public float extentV;
        }

        static Obb MakeObb(AvbdBody body)
        {
            Obb box = default;
            box.center = body.PositionLin;
            box.rotation = body.PositionAng;
            box.half = body.Size * 0.5f;
            box.axis0 = AvbdQuat.Rotate(body.PositionAng, new float3(1f, 0f, 0f));
            box.axis1 = AvbdQuat.Rotate(body.PositionAng, new float3(0f, 1f, 0f));
            box.axis2 = AvbdQuat.Rotate(body.PositionAng, new float3(0f, 0f, 1f));
            return box;
        }

        static float AbsDot(float3 a, float3 b) => math.abs(math.dot(a, b));

        static float3 SupportPoint(in Obb box, float3 dir)
        {
            float sx = math.dot(dir, box.axis0) >= 0f ? 1f : -1f;
            float sy = math.dot(dir, box.axis1) >= 0f ? 1f : -1f;
            float sz = math.dot(dir, box.axis2) >= 0f ? 1f : -1f;

            return box.center
                + box.axis0 * (box.half.x * sx)
                + box.axis1 * (box.half.y * sy)
                + box.axis2 * (box.half.z * sz);
        }

        static void GetFaceAxes(in Obb box, int axisIndex, out float3 u, out float3 v, out float extentU, out float extentV)
        {
            if (axisIndex == 0)
            {
                u = box.axis1;
                v = box.axis2;
                extentU = box.half.y;
                extentV = box.half.z;
            }
            else if (axisIndex == 1)
            {
                u = box.axis0;
                v = box.axis2;
                extentU = box.half.x;
                extentV = box.half.z;
            }
            else
            {
                u = box.axis0;
                v = box.axis1;
                extentU = box.half.x;
                extentV = box.half.y;
            }
        }

        static void BuildFaceFrame(in Obb box, int axisIndex, float3 outwardNormal, out FaceFrame frame)
        {
            float3 axis = box.Axis(axisIndex);
            float sign = math.dot(outwardNormal, axis) >= 0f ? 1f : -1f;

            frame = default;
            frame.axisIndex = axisIndex;
            frame.normal = axis * sign;
            frame.center = box.center + frame.normal * box.half[axisIndex];
            GetFaceAxes(in box, axisIndex, out frame.u, out frame.v, out frame.extentU, out frame.extentV);
        }

        static int ChooseIncidentFaceAxis(in Obb box, float3 referenceNormal)
        {
            int axis = 0;
            float best = -float.MaxValue;

            for (int i = 0; i < 3; ++i)
            {
                float d = AbsDot(box.Axis(i), referenceNormal);
                if (d > best)
                {
                    best = d;
                    axis = i;
                }
            }

            return axis;
        }

        static void BuildIncidentFace(in Obb box, int axisIndex, float3 referenceNormal, float3[] outVerts)
        {
            float3 axis = box.Axis(axisIndex);
            float sign = math.dot(axis, referenceNormal) > 0f ? -1f : 1f;
            float3 faceNormal = axis * sign;
            float3 faceCenter = box.center + faceNormal * box.half[axisIndex];

            GetFaceAxes(in box, axisIndex, out float3 u, out float3 v, out float extentU, out float extentV);

            outVerts[0] = faceCenter + u * extentU + v * extentV;
            outVerts[1] = faceCenter - u * extentU + v * extentV;
            outVerts[2] = faceCenter - u * extentU - v * extentV;
            outVerts[3] = faceCenter + u * extentU - v * extentV;
        }

        // Sutherland-Hodgman clip of a polygon against the half-space dot(planeNormal, p) <= planeOffset.
        static int ClipPolygonAgainstPlane(float3[] inVerts, int inCount, float3 planeNormal, float planeOffset, float3[] outVerts)
        {
            if (inCount <= 0)
                return 0;

            int outCount = 0;
            float3 a = inVerts[inCount - 1];
            float da = math.dot(planeNormal, a) - planeOffset;

            for (int i = 0; i < inCount; ++i)
            {
                float3 b = inVerts[i];
                float db = math.dot(planeNormal, b) - planeOffset;

                bool aInside = da <= PlaneEpsilon;
                bool bInside = db <= PlaneEpsilon;

                if (aInside != bInside)
                {
                    float t = 0f;
                    float denom = da - db;
                    if (math.abs(denom) > SatAxisEpsilon)
                        t = math.clamp(da / denom, 0f, 1f);

                    if (outCount < MaxPolyVerts)
                        outVerts[outCount++] = a + (b - a) * t;
                }

                if (bInside && outCount < MaxPolyVerts)
                    outVerts[outCount++] = b;

                a = b;
                da = db;
            }

            return outCount;
        }

        static bool AddContact(
            AvbdBody bodyA, AvbdBody bodyB, AvbdManifold.Contact[] contacts, ref int contactCount,
            float3[] contactMidpoints, float3 xA, float3 xB, int featureKey)
        {
            float3 midpoint = (xA + xB) * 0.5f;

            for (int i = 0; i < contactCount; ++i)
            {
                float3 d = midpoint - contactMidpoints[i];
                if (math.lengthsq(d) < ContactMergeDistSq)
                    return false;
            }

            if (contactCount >= MaxContacts)
                return false;

            ref AvbdManifold.Contact c = ref contacts[contactCount];
            c.featureKey = featureKey;
            c.rA = AvbdQuat.Rotate(AvbdQuat.Conjugate(bodyA.PositionAng), xA - bodyA.PositionLin);
            c.rB = AvbdQuat.Rotate(AvbdQuat.Conjugate(bodyB.PositionAng), xB - bodyB.PositionLin);
            contactMidpoints[contactCount] = midpoint;
            ++contactCount;

            return true;
        }

        static bool TestAxis(in Obb boxA, in Obb boxB, float3 delta, float3 axis, AxisType type, int indexA, int indexB, ref SatAxis best)
        {
            float lenSq = math.lengthsq(axis);
            if (lenSq < SatAxisEpsilon)
                return true;

            float invLen = 1f / math.sqrt(lenSq);
            float3 n = axis * invLen;
            if (math.dot(n, delta) < 0f)
                n = -n;

            float distance = math.abs(math.dot(delta, n));

            float rA =
                boxA.half.x * AbsDot(n, boxA.axis0) +
                boxA.half.y * AbsDot(n, boxA.axis1) +
                boxA.half.z * AbsDot(n, boxA.axis2);

            float rB =
                boxB.half.x * AbsDot(n, boxB.axis0) +
                boxB.half.y * AbsDot(n, boxB.axis1) +
                boxB.half.z * AbsDot(n, boxB.axis2);

            float separation = distance - (rA + rB);
            if (separation > 0f)
                return false;

            if (!best.valid || separation > best.separation)
            {
                best.valid = true;
                best.type = type;
                best.indexA = indexA;
                best.indexB = indexB;
                best.separation = separation;
                best.normalAB = n;
            }

            return true;
        }

        static void SupportEdge(in Obb box, int axisIndex, float3 dir, out float3 edgeA, out float3 edgeB)
        {
            int axis1 = (axisIndex + 1) % 3;
            int axis2 = (axisIndex + 2) % 3;

            float sign1 = math.dot(dir, box.Axis(axis1)) >= 0f ? 1f : -1f;
            float sign2 = math.dot(dir, box.Axis(axis2)) >= 0f ? 1f : -1f;

            float3 edgeCenter = box.center
                + box.Axis(axis1) * (box.half[axis1] * sign1)
                + box.Axis(axis2) * (box.half[axis2] * sign2);

            edgeA = edgeCenter - box.Axis(axisIndex) * box.half[axisIndex];
            edgeB = edgeCenter + box.Axis(axisIndex) * box.half[axisIndex];
        }

        static void ClosestPointsOnSegments(float3 p0, float3 p1, float3 q0, float3 q1, out float3 c0, out float3 c1)
        {
            float3 d1 = p1 - p0;
            float3 d2 = q1 - q0;
            float3 r = p0 - q0;
            float a = math.dot(d1, d1);
            float e = math.dot(d2, d2);
            float f = math.dot(d2, r);

            float s = 0f;
            float t = 0f;

            if (a <= SatAxisEpsilon && e <= SatAxisEpsilon)
            {
                c0 = p0;
                c1 = q0;
                return;
            }

            if (a <= SatAxisEpsilon)
            {
                t = math.clamp(f / e, 0f, 1f);
            }
            else
            {
                float c = math.dot(d1, r);
                if (e <= SatAxisEpsilon)
                {
                    s = math.clamp(-c / a, 0f, 1f);
                }
                else
                {
                    float b = math.dot(d1, d2);
                    float denom = a * e - b * b;

                    if (math.abs(denom) > SatAxisEpsilon)
                        s = math.clamp((b * f - c * e) / denom, 0f, 1f);

                    t = (b * s + f) / e;

                    if (t < 0f)
                    {
                        t = 0f;
                        s = math.clamp(-c / a, 0f, 1f);
                    }
                    else if (t > 1f)
                    {
                        t = 1f;
                        s = math.clamp((b - c) / a, 0f, 1f);
                    }
                }
            }

            c0 = p0 + d1 * s;
            c1 = q0 + d2 * t;
        }

        static int BuildFaceManifold(
            AvbdBody bodyA, AvbdBody bodyB, in Obb boxA, in Obb boxB, bool referenceIsA, int referenceAxis,
            float3 normalAB, AvbdManifold.Contact[] contacts)
        {
            Obb referenceBox = referenceIsA ? boxA : boxB;
            Obb incidentBox = referenceIsA ? boxB : boxA;
            float3 referenceOutward = referenceIsA ? normalAB : -normalAB;

            BuildFaceFrame(in referenceBox, referenceAxis, referenceOutward, out FaceFrame referenceFace);

            int incidentAxis = ChooseIncidentFaceAxis(in incidentBox, referenceFace.normal);

            float3[] clip0 = new float3[MaxPolyVerts];
            float3[] clip1 = new float3[MaxPolyVerts];
            BuildIncidentFace(in incidentBox, incidentAxis, referenceFace.normal, clip0);
            int count = 4;

            // Clip the incident face against the four side planes of the reference face.
            float3 n0 = referenceFace.u;
            float o0 = math.dot(n0, referenceFace.center) + referenceFace.extentU;
            count = ClipPolygonAgainstPlane(clip0, count, n0, o0, clip1);
            if (count == 0)
                return 0;

            float3 n1 = -referenceFace.u;
            float o1 = math.dot(n1, referenceFace.center) + referenceFace.extentU;
            count = ClipPolygonAgainstPlane(clip1, count, n1, o1, clip0);
            if (count == 0)
                return 0;

            float3 n2 = referenceFace.v;
            float o2 = math.dot(n2, referenceFace.center) + referenceFace.extentV;
            count = ClipPolygonAgainstPlane(clip0, count, n2, o2, clip1);
            if (count == 0)
                return 0;

            float3 n3 = -referenceFace.v;
            float o3 = math.dot(n3, referenceFace.center) + referenceFace.extentV;
            count = ClipPolygonAgainstPlane(clip1, count, n3, o3, clip0);
            if (count == 0)
                return 0;

            int contactCount = 0;
            float3[] contactMidpoints = new float3[MaxContacts];
            int featurePrefix = (int)(referenceIsA ? AxisType.FaceA : AxisType.FaceB) << 24;
            featurePrefix |= (referenceAxis & 0xFF) << 16;
            featurePrefix |= (incidentAxis & 0xFF) << 8;

            for (int i = 0; i < count && contactCount < MaxContacts; ++i)
            {
                float3 pIncident = clip0[i];
                float distance = math.dot(pIncident - referenceFace.center, referenceFace.normal);
                if (distance > PlaneEpsilon)
                    continue;

                float3 pReference = pIncident - referenceFace.normal * distance;
                float3 xA = referenceIsA ? pReference : pIncident;
                float3 xB = referenceIsA ? pIncident : pReference;

                AddContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featurePrefix | (i & 0xFF));
            }

            if (contactCount == 0)
            {
                float3 xA = SupportPoint(in boxA, normalAB);
                float3 xB = SupportPoint(in boxB, -normalAB);
                AddContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featurePrefix);
            }

            return contactCount;
        }

        static int BuildEdgeContact(
            AvbdBody bodyA, AvbdBody bodyB, in Obb boxA, in Obb boxB, int axisA, int axisB,
            float3 normalAB, AvbdManifold.Contact[] contacts)
        {
            SupportEdge(in boxA, axisA, normalAB, out float3 a0, out float3 a1);
            SupportEdge(in boxB, axisB, -normalAB, out float3 b0, out float3 b1);

            ClosestPointsOnSegments(a0, a1, b0, b1, out float3 xA, out float3 xB);

            int contactCount = 0;
            float3[] contactMidpoints = new float3[MaxContacts];
            int featureKey = ((int)AxisType.Edge << 24) | ((axisA & 0xFF) << 8) | (axisB & 0xFF);
            AddContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featureKey);

            if (contactCount == 0)
            {
                xA = SupportPoint(in boxA, normalAB);
                xB = SupportPoint(in boxB, -normalAB);
                AddContact(bodyA, bodyB, contacts, ref contactCount, contactMidpoints, xA, xB, featureKey);
            }

            return contactCount;
        }

        /// <summary>
        /// Collides two boxes. Writes up to <see cref="MaxContacts"/> contacts (feature key, rA, rB only;
        /// the rest of each contact is left as it was) into <paramref name="contacts"/> and returns their
        /// number. <paramref name="basis"/> is overwritten with the contact basis (row 0 = normal pointing
        /// from B to A, rows 1 and 2 = tangents) whenever the boxes overlap on all axes, and left
        /// unchanged when the separating-axis test finds a gap, matching upstream.
        /// </summary>
        public static int Collide(AvbdBody bodyA, AvbdBody bodyB, AvbdManifold.Contact[] contacts, ref float3x3 basis)
        {
            Obb boxA = MakeObb(bodyA);
            Obb boxB = MakeObb(bodyB);
            float3 delta = boxB.center - boxA.center;

            SatAxis bestFace = default;
            bestFace.separation = -float.MaxValue;
            bestFace.valid = false;

            SatAxis bestEdge = default;
            bestEdge.separation = -float.MaxValue;
            bestEdge.valid = false;

            for (int i = 0; i < 3; ++i)
            {
                if (!TestAxis(in boxA, in boxB, delta, boxA.Axis(i), AxisType.FaceA, i, -1, ref bestFace))
                    return 0;
            }

            for (int i = 0; i < 3; ++i)
            {
                if (!TestAxis(in boxA, in boxB, delta, boxB.Axis(i), AxisType.FaceB, -1, i, ref bestFace))
                    return 0;
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    float3 axis = math.cross(boxA.Axis(i), boxB.Axis(j));
                    if (!TestAxis(in boxA, in boxB, delta, axis, AxisType.Edge, i, j, ref bestEdge))
                        return 0;
                }
            }

            if (!bestFace.valid)
                return 0;

            SatAxis best = bestFace;
            if (bestEdge.valid)
            {
                const float edgeRelTol = 0.95f;
                const float edgeAbsTol = 0.01f;
                if (edgeRelTol * bestEdge.separation > bestFace.separation + edgeAbsTol)
                    best = bestEdge;
            }

            basis = Orthonormal(-best.normalAB);

            if (best.type == AxisType.Edge)
                return BuildEdgeContact(bodyA, bodyB, in boxA, in boxB, best.indexA, best.indexB, best.normalAB, contacts);

            if (best.type == AxisType.FaceA)
                return BuildFaceManifold(bodyA, bodyB, in boxA, in boxB, true, best.indexA, best.normalAB, contacts);

            return BuildFaceManifold(bodyA, bodyB, in boxA, in boxB, false, best.indexB, best.normalAB, contacts);
        }

        // Rows: the normal, then two tangents (upstream's orthonormal()).
        static float3x3 Orthonormal(float3 normal)
        {
            float3 t1 = math.abs(normal.x) > math.abs(normal.z)
                ? new float3(-normal.y, normal.x, 0f)
                : new float3(0f, -normal.z, normal.y);
            t1 = math.normalize(t1);
            float3 t2 = math.cross(normal, t1);
            return AvbdMat3.FromRows(normal, t1, t2);
        }
    }
}
