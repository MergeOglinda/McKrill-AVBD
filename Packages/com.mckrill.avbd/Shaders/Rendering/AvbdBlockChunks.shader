// Draws the smooth surface pieces ("chunks") of blocks straight from the solver's body state buffer, in one draw call for all blocks.
//
// A chunk is a closed piece of the smooth surface of an object that belongs to one block (see Tools/Voxelize/smooth_chunks.py). All chunks are in one
// mesh. Every vertex has its position and normal relative to the center of its block, and the number of its block (a 32-bit integer in TEXCOORD0). The
// vertex shader reads _Instances[block] (the body slot and the color of the block, the same 32 B record as AvbdInstancedBoxes; the sizes are not used)
// and then _Poses[slot] (the pose the solver wrote), and moves the vertex rigidly with the body: rotation, then translation. A block that breaks off an
// object takes its chunk with it. Draw it with Graphics.RenderMeshPrimitives(params, mesh, 0) and bind the two buffers on the material.
//
// The lighting is the one of AvbdInstancedBoxes (one fixed directional light plus ambient, no shadows), so chunks and plain boxes match. The colors in
// _Instances are the final colors: convert them to linear first when the project uses the linear color space.
Shader "McKrill/AVBD Block Chunks"
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

            // The first two float4 of the 64 B body state record (the other two are not read here).
            struct Pose
            {
                float4 position;
                float4 rotation; // x, y, z, w
                float4 startPosition;
                float4 startRotation;
            };

            // 32 B, as in AvbdInstancedBoxes.
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

            Varyings Vert(float3 vertex : POSITION, float3 normal : NORMAL, uint block : TEXCOORD0)
            {
                Instance instance = _Instances[block];
                Pose pose = _Poses[instance.slot];

                float3 world = Rotate(pose.rotation, vertex) + pose.position.xyz;

                Varyings o;
                o.position = mul(UNITY_MATRIX_VP, float4(world, 1.0));
                o.normal = Rotate(pose.rotation, normal); // a rigid motion: the normal needs only the rotation
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
