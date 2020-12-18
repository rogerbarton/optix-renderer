//
// Created by roger on 06/12/2020.
//
#include <cuda_runtime.h>
#include <optix_device.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "../cuda_shared/LaunchParams.h"
#include "../cuda_shared/RayParams.h"

#include "RadiancePrd.h"

#include "shaders/camera.h"

#include "sutil/exception.h"
#include "sutil/helpers.h"
#include "sutil/vec_math.h"
#include "sutil/random.h"
#include "sutil/warp.h"

extern "C" {
__constant__ LaunchParams launchParams;
}

static __forceinline__ __device__ void traceRadiance(
		OptixTraversableHandle handle,
		float3 ray_origin,
		float3 ray_direction,
		float tmin,
		float tmax,
		RadiancePrd *prd)
{
	unsigned int u0, u1;
	packPointer(prd, u0, u1);
	optixTrace(
			handle,
			ray_origin,
			ray_direction,
			tmin,
			tmax,
			0.0f,                        // rayTime
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_NONE,
			RAY_TYPE_RADIANCE,          // SBT offset
			RAY_TYPE_COUNT,             // SBT stride
			RAY_TYPE_RADIANCE,       // missSBTIndex
			u0, u1);
}

static __forceinline__ __device__ bool traceShadowray(
		OptixTraversableHandle handle,
		float3 ray_origin,
		float3 ray_direction,
		float tmin,
		float tmax)
{
	unsigned int occluded = 0u;
	optixTrace(
			handle,
			ray_origin,
			ray_direction,
			tmin,
			tmax,
			0.0f,                        // rayTime
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_NONE,
			RAY_TYPE_SHADOWRAY,         // SBT offset
			RAY_TYPE_COUNT,             // SBT stride
			RAY_TYPE_SHADOWRAY,      // missSBTIndex
			occluded);
	return occluded;
}

extern "C" __global__ void __raygen__perspective()
{
	const unsigned int sampleIndex = launchParams.sampleIndex;
	const uint3        idx         = optixGetLaunchIndex();
	const unsigned int pixelIdx    = idx.y * launchParams.imageWidth + idx.x;

	unsigned int seed = tea<4>(pixelIdx, sampleIndex);

	float3 color  = {0, 0, 0};
	float3 albedo = {0, 0, 0};
	float3 normal = {0, 0, 0};

	for (int i = 0; i < launchParams.samplesPerLaunch; ++i)
	{
		float3 rayDirection;
		float3 rayOrigin;
		sampleRay(launchParams, seed, rayOrigin, rayDirection);

		// -- Integrator::Li()
		RadiancePrd prd{};
		prd.Li         = make_float3(0);
		prd.throughput = make_float3(1);
		prd.albedo     = make_float3(-1);
		prd.normal     = -rayDirection;
		prd.terminated = false;
		prd.seed       = seed;

		for (int depth = 0;; ++depth)
		{
			traceRadiance(
					launchParams.sceneHandle,
					rayOrigin,
					rayDirection,
					EPSILON,
					INFINITY,
					&prd);

			if (depth == 0)
				normal += prd.normal;

			if (prd.terminated)
				break;

			// russian roulette
			if (depth >= 3)
			{
				const float rouletteSuccess = fminf(fmaxf3(prd.throughput), 0.99f);

				if (rnd(prd.seed) > rouletteSuccess || rouletteSuccess < EPSILON)
					break;
				// Adjust throughput in case of survival
				prd.throughput /= rouletteSuccess;
			}

			rayDirection = prd.direction;
			rayOrigin    = prd.origin;
		}

		seed = prd.seed;

		// Store Li
		color += prd.Li;
		albedo += prd.albedo;
	}

	// Normalize and update image
	interpolateAndApplyToBuffer(launchParams.sampleIndex, launchParams.samplesPerLaunch, pixelIdx,
	                            launchParams.d_outputBuffer, color);
	interpolateAndApplyToBuffer(launchParams.sampleIndex, launchParams.samplesPerLaunch, pixelIdx,
	                            launchParams.d_outputBufferAlbedo, albedo);
	interpolateAndApplyToBuffer(launchParams.sampleIndex, launchParams.samplesPerLaunch, pixelIdx,
	                            launchParams.d_outputBufferNormal, normal);
}
