//
// Created by roger on 09/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_device.h>
#include <nori/optix/cuda_shared/RayParams.h>
#include <nori/optix/cuda_shared/LaunchParams.h>
#include <nori/optix/cuda_shared/EmitterData.h>

#include "../sutil/helpers.h"
#include "../sutil/random.h"
#include "../sutil/warp.h"
#include "../sutil/exception.h"
#include "../sutil/shader_helpers.h"

/**
 * Emitter::eval, return Li
 * Emitter should valid, i.e. not type == NONE
 */
static __forceinline__ __device__ float3 evalEmitter(
		const HitGroupParams &sbtData, const float3 &ref, const float3 &p, const float3 &n)
{
	const EmitterData &emitter = sbtData.emitter;
	if (emitter.type == EmitterData::POINT)
	{
		return emitter.radiance / normSqr(ref - emitter.point.position);
	}
	else if (emitter.type == EmitterData::SPOT)
	{
		// TODO: implement falloff(), efficiency...
		// float cosFalloffStart = std::cos(emitter.spot.falloffStart);
		// float cosTotalWidth = std::cos(emitter.spot.totalWidthAngle / 2.f);
		// float3 i = emitter.radiance / (1.f - .5f * (cosTotalWidth + cosFalloffStart));
		// float3 color = i * normSqr(falloff(-wi) / (ref - emitter.spot.position));
		// return color;
	}
	else if (emitter.type == EmitterData::AREA)
	{
		return emitter.radiance;
	}
	else if (emitter.type == EmitterData::VOLUME)
	{
		return emitter.radiance;
	}

	return ERROR_COLOR;
}


/**
 * Emitter::sample, return weighted Li
 * Emitter should valid, i.e. not type == NONE
 */
static __forceinline__ __device__ float3 sampleEmitter(
		const HitGroupParams &sbtData, const float3 &ref,
		float3 &p, float3 &n, float &pdf, unsigned int &seed)
{
	const EmitterData &emitter = sbtData.emitter;
	if (emitter.type == EmitterData::POINT)
	{
		p = emitter.point.position;
		n = make_float3(0.f);
		return emitter.radiance / normSqr(ref - emitter.point.position);
	}
	else if (emitter.type == EmitterData::SPOT)
	{

	}
	else if (emitter.type == EmitterData::AREA)
	{

	}
	else if (emitter.type == EmitterData::VOLUME)
	{

	}

	return ERROR_COLOR;
}