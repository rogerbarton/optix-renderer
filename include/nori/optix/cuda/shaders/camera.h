//
// Created by roger on 16/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_device.h>
#include <nori/optix/cuda/sutil/exception.h>
#include "../../cuda_shared/LaunchParams.h"
#include "../sutil/helpers.h"
#include "../sutil/random.h"
#include "../sutil/warp.h"

/**
 * Interface function that implements the different camera models for sampling rays:
 * 1. Pinhole (partially tested, may be inverted)
 * 2. Perspective (based on nori version)
 */
__forceinline__ __device__ void sampleRay(
		const LaunchParams &launchParams,
		unsigned int &seed,
		float3 &rayOrigin,
		float3 &rayDirection);


/**
 * Apply a 4x4 transformation matrix using homogeneous coordinates
 * @param tx first row
 * @param ty second row
 * @param p 3d point to transform
 * @return Transformed 3d point
 */
__forceinline__ __device__ float3 apply4x4Transform(float4 tx, float4 ty, float4 tz, float4 tw, float3 p)
{
	const float4 ph     = make_float4(p.x, p.y, p.z, 1.f);
	// Apply matrix in homogeneous coords
	float4       result = make_float4(
			dot(tx, ph),
			dot(ty, ph),
			dot(tz, ph),
			dot(tw, ph));
	// Convert back
	return make_float3(result.x, result.y, result.z) / result.w;
}

__forceinline__ __device__ float3 apply4x4TransformRotation(float4 tx, float4 ty, float4 tz, float4 tw, float3 p)
{
	return make_float3(tx.x * p.x + tx.y * p.y + tx.z * p.z,
	                   ty.x * p.x + ty.y * p.y + ty.z * p.z,
	                   tz.x * p.x + tz.y * p.y + tz.z * p.z);
}

__forceinline__ __device__ void sampleRayPinhole(
		const LaunchParams &launchParams, unsigned int &seed, float3 &rayOrigin, float3 &rayDirection)
{
	const uint3               idx   = optixGetLaunchIndex();
	const unsigned int        w     = launchParams.imageWidth;
	const unsigned int        h     = launchParams.imageHeight;
	const RaygenData::Pinhole &data = launchParams.camera.pinhole;
	const float3              eye   = data.eye;
	const float3              U     = data.U;
	const float3              V     = data.V;
	const float3              W     = data.W;

	const float2 subpixelJitter = make_float2(rnd(seed), rnd(seed));

	// Local ray
	const float2 d = 2.0f * make_float2(
			(static_cast<float>(idx.x) + subpixelJitter.x) / static_cast<float>(w),
			(static_cast<float>(idx.y) + subpixelJitter.y) / static_cast<float>(h)
	) - 1.0f;

	// Transform camera to world
	rayOrigin    = eye;
	rayDirection = normalize(d.x * U + d.y * V + W);
}

__forceinline__ __device__ static void sampleRayPerspective(
		const LaunchParams &launchParams, unsigned int &seed, float3 &rayOrigin, float3 &rayDirection)
{
	const uint3                   idx   = optixGetLaunchIndex();
	const RaygenData::Perspective &data = launchParams.camera.perspective;

	const float2 subpixelJitter = make_float2(rnd(seed), rnd(seed));

	const float2 screen         = make_float2(
			(static_cast<float>(idx.x) + subpixelJitter.x),
			(static_cast<float>(idx.y) + subpixelJitter.y));

	const float3 viewport = make_float3(screen.x * data.invOutputSize.x, screen.y * data.invOutputSize.y, 0);

	const float3 nearP = apply4x4Transform(data.sampleToCameraX,
	                                       data.sampleToCameraY,
	                                       data.sampleToCameraZ,
	                                       data.sampleToCameraW, viewport);

	// Create local space ray
	float3 d = normalize(nearP);
	float3 o = make_float3(0, 0, 0);

	// Depth of field, adjusts ray in local space
	if (data.lensRadius > EPSILON)
	{
		const float2 pLens = data.lensRadius * squareToUniformDisk(make_float2(rnd(seed), rnd(seed)));
		const float  ft    = data.focalDistance / d.z;

		// position of ray at time of intersection with the focal plane
		const float3 pFocus = o + d * ft;

		o = make_float3(pLens.x, pLens.y, 0.f);
		// direction connecting aperture and focal plane points
		d = normalize(pFocus - o);
	}

	// To world space
	rayOrigin    = apply4x4Transform(data.cameraToWorldX,
	                                 data.cameraToWorldY,
	                                 data.cameraToWorldZ,
	                                 data.cameraToWorldW, o);
	rayDirection = apply4x4TransformRotation(data.cameraToWorldX,
	                                         data.cameraToWorldY,
	                                         data.cameraToWorldZ,
	                                         data.cameraToWorldW, d);

	rayDirection = normalize(rayDirection);
}

__forceinline__ __device__ void sampleRay(
		const LaunchParams &launchParams,
		unsigned int &seed,
		float3 &rayOrigin,
		float3 &rayDirection)
{
	switch (launchParams.camera.type)
	{
		case RaygenData::PINHOLE:
			sampleRayPinhole(launchParams, seed, rayOrigin, rayDirection);
			break;
		case RaygenData::PERSPECTIVE:
			sampleRayPerspective(launchParams, seed, rayOrigin, rayDirection);
			break;
	}

	// PRINT_PIXEL(192, 192, "ray.d  (%f, %f, %f) \n", rayDirection.x, rayDirection.y, rayDirection.z);
}
