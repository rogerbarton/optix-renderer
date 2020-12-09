//
// Created by roger on 09/12/2020.
//

#pragma once

#include <cuda_runtime.h>

/**
 * Per ray data (PRD) for the radiance ray.
 * Keep this 16-bit aligned (?) by adding appropriate padding
 */
struct RadiancePrd
{
	float3       Li;
	float3       throughput;
	float3       origin;
	float3       direction;
	unsigned int seed;
	int          terminated;
	int          pad[2];
};