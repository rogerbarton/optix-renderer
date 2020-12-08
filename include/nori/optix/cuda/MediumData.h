//
// Created by roger on 08/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>
#include <vector_types.h>

struct MediumData
{
	enum Type
	{
		VACUUM = 0,
		HOMOG  = 1,
		HETROG = 2,
		TYPE_COUNT
	};

	struct Vacuum
	{
	};

	struct Homog
	{
		float3 mu_a;
		float3 mu_s;
		float3 mu_t;
		float  density;
	};

	struct Hetrog
	{
	};

	Type type = VACUUM;

	union
	{
		Vacuum vacuum;
		Homog  homog;
		Hetrog hetrog;
	};
};