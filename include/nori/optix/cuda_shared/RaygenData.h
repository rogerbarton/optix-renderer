//
// Created by roger on 16/12/2020.
//

#pragma once

#include <optix_types.h>
#include <vector_types.h>

struct RaygenData
{
	enum Type
	{
		PINHOLE     = 0,
		PERSPECTIVE = 1,
		TYPE_SIZE
	};

	struct Pinhole
	{
		float  fovY;
		float3 eye;
		float3 U;
		float3 V;
		float3 W;
	};

	struct Perspective
	{
		float nearClip;
		float farClip;
		float focalDistance;
		float lensRadius;
		float2 invOutputSize;

		// 4x4 Transformations with rows [X;Y;Z;W]
		float4 sampleToCameraX;
		float4 sampleToCameraY;
		float4 sampleToCameraZ;
		float4 sampleToCameraW;
		float4 cameraToWorldX;
		float4 cameraToWorldY;
		float4 cameraToWorldZ;
		float4 cameraToWorldW;
	};

	Type type;

	union
	{
		Pinhole     pinhole;
		Perspective perspective;
	};
};
