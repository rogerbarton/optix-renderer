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

#include "sutil/helpers.h"
#include "sutil/vec_math.h"
#include "sutil/random.h"
#include "sutil/warp.h"

static __forceinline__ __device__ void sampleRay(
		unsigned int &seed,
		float3 &ray_direction,
		float3 &ray_origin);

static __forceinline__ __device__ void traceRadiance(
		OptixTraversableHandle handle,
		float3 ray_origin,
		float3 ray_direction,
		float tmin,
		float tmax,
		RadiancePrd *prd);

static __forceinline__ __device__ bool traceShadowray(
		OptixTraversableHandle handle,
		float3 ray_origin,
		float3 ray_direction,
		float tmin,
		float tmax);

extern "C" {
__constant__ LaunchParams launchParams;
}

extern "C" __global__ void __raygen__perspective()
{
	const unsigned int sampleIndex = launchParams.sampleIndex;
	const uint3        idx         = optixGetLaunchIndex();
	const unsigned int pixelIdx    = idx.y * launchParams.imageWidth + idx.x;

	unsigned int seed = tea<4>(pixelIdx, sampleIndex);

	float3 color = {0, 0, 0};

	for (int i = 0; i < launchParams.samplesPerLaunch; ++i)
	{
		float3 rayDirection;
		float3 rayOrigin;
		sampleRay(seed, rayDirection, rayOrigin);

		// -- Integrator::Li()
		RadiancePrd prd{};
		prd.Li         = make_float3(0.f);
		prd.throughput = make_float3(1.f);
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

			if (prd.terminated)
				break;

			// russian roulette
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
	}

	// Normalize and update image
	color /= static_cast<float>(launchParams.samplesPerLaunch);

	if (launchParams.sampleIndex > 1)
	{
		float3 prevPixel = make_float3(launchParams.d_imageBuffer[pixelIdx]);

		const float a = 1.0f / static_cast<float>(launchParams.sampleIndex + 1);
		color = lerp(prevPixel, color, a);
	}

	launchParams.d_imageBuffer[pixelIdx] = make_float4(color, 1.f);
}

static __forceinline__ __device__ void sampleRay(
		unsigned int &seed,
		float3 &ray_direction,
		float3 &ray_origin)
{
	const unsigned int w   = launchParams.imageWidth;
	const unsigned int h   = launchParams.imageHeight;
	const float3       eye = launchParams.camera.eye;
	const float3       U   = launchParams.camera.U;
	const float3       V   = launchParams.camera.V;
	const float3       W   = launchParams.camera.W;
	const uint3        idx = optixGetLaunchIndex();

	const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));

	// Local ray
	float3 o = {0, 0, 0};
	float3 d = 2.0f * make_float3(
			(static_cast<float>( idx.x ) + subpixel_jitter.x) / static_cast<float>( w ),
			(static_cast<float>( idx.y ) + subpixel_jitter.y) / static_cast<float>( h ),
			0.5f
	) - 1.0f;

	if (launchParams.camera.lensRadius > EPSILON)
	{
		const float2 pLens  = launchParams.camera.lensRadius *
		                      squareToUniformDisk(make_float2(rnd(seed), rnd(seed)));
		const float  ft     = launchParams.camera.focalDistance / d.z;
		const float3 pFocus = o + d * ft;

		o = make_float3(pLens.x, pLens.y, 0.f);
		// direction connecting aperture and focal plane points
		d = normalize(pFocus - o);
	}

	// Transform camera to world
	ray_direction = normalize(d.x * U + d.y * V + d.z * W);
	ray_origin    = eye + o.x * U + o.y * V + o.z * W;
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
			OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
			RAY_TYPE_SHADOWRAY,         // SBT offset
			RAY_TYPE_COUNT,             // SBT stride
			RAY_TYPE_SHADOWRAY,      // missSBTIndex
			occluded);
	return occluded;
}
