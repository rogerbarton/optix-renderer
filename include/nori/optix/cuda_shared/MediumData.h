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
		VACUUM  = 0,
		HOMOG   = 1,
		HETEROG = 2,
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
		float  density; // TODO: do we need this? albedo?
	};

	struct Heterog
	{
		const void *densityGrid;
		const void *temperatureGrid;
	};

	Type type = VACUUM;

	union
	{
		Vacuum  vacuum;
		Homog   homog;
		Heterog heterog;
	};
};