//
// Created by roger on 06/12/2020.
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"

#include <cuda_runtime.h>
#include <optix_device.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "LaunchParams.h"
#include "RayParams.h"
#include "sutil/helpers.h"

#include <nanovdb/util/Ray.h>


extern "C" {
__constant__ LaunchParams launchParams;
}

extern "C" __global__ void __raygen__perspective()
{
	using RealT = float;
	using Vec3T = nanovdb::Vec3<RealT>;
	using CoordT = nanovdb::Coord;
	using RayT = nanovdb::Ray<RealT>;

	const uint3    idx          = optixGetLaunchIndex();
	const uint3    dim          = optixGetLaunchDimensions();
	const uint32_t ix           = idx.x;
	const uint32_t iy           = idx.y;
	const uint32_t offset       = launchParams.imageWidth * idx.y + idx.x;
	const auto     &sceneParams = launchParams.scene;

	float3 color = {0, 0, 0};

	for (int sampleIndex = 0; sampleIndex < launchParams.samplesPerLaunch; ++sampleIndex)
	{
		uint32_t pixelSeed = render::hash(
				(sampleIndex + (launchParams.sampleIndex + 1) * launchParams.samplesPerLaunch)) ^
		                     render::hash(ix, iy);

		RayT wRay = render::getRayFromPixelCoord(ix, iy, launchParams.imageWidth, launchParams.imageHeight,
		                                         launchParams.sampleIndex, launchParams.samplesPerLaunch, pixelSeed,
		                                         sceneParams);

		float3 result;
		optixTrace(
				launchParams.sceneHandle,
				make_float3(wRay.eye()),
				make_float3(wRay.dir()),
				Epsilon,
				1e16f,
				0.0f,
				OptixVisibilityMask(1),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				RAY_TYPE_RADIANCE,
				RAY_TYPE_COUNT,
				RAY_TYPE_RADIANCE,
				float3_as_args(result));

		color += result;
	}

	color /= (float) launchParams.samplesPerLaunch;

	if (launchParams.sampleIndex > 1)
	{
		float3 prevPixel = make_float3(launchParams.d_imageBuffer[offset]);

		color = prevPixel + (color - prevPixel) * (1.0f / launchParams.sampleIndex);
	}

	launchParams.d_imageBuffer[offset] = make_float4(color, 1.f);
}

__forceinline__ float3 Li()
{

}
#pragma clang diagnostic pop