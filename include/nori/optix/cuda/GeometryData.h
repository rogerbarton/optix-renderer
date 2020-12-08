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
		TRIANGLE_MESH         = 0,
		SPHERE                = 1,
	};

	struct TriangleMesh
	{
		float3  *positions;
		float3  *normals;
		float2  *texcoords;
		int32_t *indices;
	};

	struct Sphere
	{
		float3 center;
		float  radius;
	};

	Type type;

	union
	{
		TriangleMesh triangleMesh;
		Sphere       sphere;
	};
};