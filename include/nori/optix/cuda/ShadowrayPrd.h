//
// Created by roger on 09/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_device.h>
#include "sutil/helpers.h"

/**
 * Per ray data (PRD) for the shadow ray.
 * This struct is mapped directly to the registers
 */
struct ShadowrayPrd
{
	/**
	 * Mapping of the 8 available payload registers
	 * Payload 0: bool occluded
	 */

	static __forceinline__ __device__ void setPrd(bool occluded)
	{
		optixSetPayload_0(static_cast<unsigned int>(occluded));
	}

	static __forceinline__ __device__ void getPrd(bool& occluded)
	{
		occluded = optixGetPayload_0();
	}
};
