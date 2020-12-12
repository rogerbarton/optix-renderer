//
// Created by roger on 09/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_device.h>
#include "sutil/helpers.h"

/**
 * Per ray data (PRD) for the radiance ray.
 * Keep this 16-bit aligned (?) by adding appropriate padding
 */
struct RadiancePrd
{
	/**
	 * Mapping of the 8 available payload registers
	 * Payload 0: RadiancePrd pointer part 1
	 * Payload 1: RadiancePrd pointer part 2
	 * Note: As only a const data pointer is used in the payload, we do not require a setter.
	 * TODO: map seed to register because of frequent usage
	 */

	float3       Li;
	float3       throughput;
	float3       albedo;
	float3       normal;
	float3       origin;
	float3       direction;
	unsigned int seed;
	int          terminated;
	// int          pad;

	static __forceinline__ __device__ RadiancePrd* getPrd()
	{
		const unsigned int u0 = optixGetPayload_0();
		const unsigned int u1 = optixGetPayload_1();
		return reinterpret_cast<RadiancePrd*>( unpackPointer( u0, u1 ) );
	}
};

