// Draws boxes straight from the solver's body state buffer: one instance per box, no readback, no per-box object.
//
// The vertex shader reads, for instance i, _Instances[i] (the body slot, the size of the box and its color) and then _Poses[slot] (the pose the
// solver wrote: position and rotation, the first two float4 of a body state record, see the design doc, section 3). The mesh is a unit cube
// (-0.5 to 0.5). Draw it with Graphics.RenderMeshPrimitives(params, mesh, 0, instanceCount) and bind the two buffers on the material.
//
// The shader is plain unlit-style code with one fixed directional light plus ambient, so it renders the same in the built-in pipeline and in URP
// (a pass without a LightMode tag is drawn by both). It has no shadow pass and receives no shadows. The colors in _Instances are the final
// colors: convert them to linear first when the project uses the linear color space. The mesh can be any mesh of diameter 1 (a unit sphere, say): the
// size scales it. A float _InstanceOffset on the material starts the draw at that instance.
Shader "McKrill/AVBD Instanced Boxes"
{
    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Geometry" }

        Pass
        {
            Cull Back
            ZWrite On
            ZTest LEqual

            CGPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag

            #include "UnityCG.cginc"

            // The first two float4 of the 64 B body state record (the other two, the pose at the start of the step, are not read here).
            struct Pose
            {
                float4 position;
                float4 rotation; // x, y, z, w
                float4 startPosition;
                float4 startRotation;
            };

            // 32 B: only 4-byte scalars before the float4, so the layout is the same in every graphics API.
            struct Instance
            {
                uint slot;
                float sizeX;
                float sizeY;
                float sizeZ;
                float4 color;
            };

            StructuredBuffer<Pose> _Poses;
            StructuredBuffer<Instance> _Instances;

            // The number of the first instance of the draw: a material that sets it draws a range of _Instances (the buffer holds the ground, the voxels and the
            // shots, say, and the voxels are spheres drawn with another mesh). 0 when the property is not set: the instances from the start.
            float _InstanceOffset;

            float4 _LightDirection; // xyz: unit vector towards the light
            float4 _LightColor;
            float4 _AmbientColor;

            struct Varyings
            {
                float4 position : SV_POSITION;
                float3 normal : TEXCOORD0;
                float4 color : COLOR0;
            };

            float3 Rotate(float4 q, float3 v)
            {
                return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
            }

            Varyings Vert(float4 vertex : POSITION, float3 normal : NORMAL, uint instanceId : SV_InstanceID)
            {
                Instance instance = _Instances[instanceId + (uint)_InstanceOffset];
                Pose pose = _Poses[instance.slot];

                float3 local = vertex.xyz * float3(instance.sizeX, instance.sizeY, instance.sizeZ);
                float3 world = Rotate(pose.rotation, local) + pose.position.xyz;

                Varyings o;
                o.position = mul(UNITY_MATRIX_VP, float4(world, 1.0));
                // The faces of a box keep their directions under a non-uniform scale, so the normal needs only the rotation.
                o.normal = Rotate(pose.rotation, normal);
                o.color = instance.color;
                return o;
            }

            float4 Frag(Varyings i) : SV_Target
            {
                float3 n = normalize(i.normal);
                float3 light = _AmbientColor.rgb * (0.75 + 0.25 * n.y) + _LightColor.rgb * saturate(dot(n, _LightDirection.xyz));
                return float4(i.color.rgb * light, 1.0);
            }
            ENDCG
        }
    }

    Fallback Off
}
