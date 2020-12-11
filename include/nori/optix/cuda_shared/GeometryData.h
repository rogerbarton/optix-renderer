//
// Created by roger on 08/12/2020.
//

/**
 * Based on OptiX samples
 */
#pragma once

#include <optix_types.h>
#include <vector_types.h>

struct GeometryData
{
	enum Type
	{
		TRIANGLE_MESH = 0,
		SPHERE        = 1,
		VOLUME_NVDB   = 2,
		TYPE_SIZE
	};

	struct TriangleMesh
	{
		float3 *V;
		float3 *N;
		float2 *UV;
		uint3  *F;
	};

	struct Sphere
	{
		float3 center;
		float  radius;
	};

	struct VolumeNvdb
	{
		const void* grid;
	};

	Type type;
	float volume;

	union
	{
		TriangleMesh triangleMesh;
		Sphere       sphere;
		VolumeNvdb   volumeNvdb;
	};
};