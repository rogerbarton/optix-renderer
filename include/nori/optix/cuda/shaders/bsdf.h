//
// Created by roger on 09/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_device.h>
#include <nori/optix/cuda_shared/BsdfData.h>

#include "../sutil/helpers.h"
#include "../sutil/random.h"
#include "../sutil/warp.h"
#include "../sutil/exception.h"
#include "../sutil/shader_helpers.h"


/**
 * Bsdf::eval
 */
static __forceinline__ __device__ float3 evalBsdf(
		const BsdfData &bsdf, const float2 &uv, const float3 &wi, const float3 &wo)
{
	float cosTheta = wi.z;

	if (bsdf.type == BsdfData::DIFFUSE)
	{
		if (bsdf.diffuse.albedoTex == 0)
			return M_1_PIf * bsdf.diffuse.albedo;
		else
			return M_1_PIf * make_float3(tex2D<float4>(bsdf.diffuse.albedoTex, uv.x, uv.y));
	}
	else if (bsdf.type == BsdfData::MIRROR)
	{
		return make_float3(0.f);
	}
	else if (bsdf.type == BsdfData::DIELECTRIC)
	{
		return make_float3(0.f);
	}
	else if (bsdf.type == BsdfData::MICROFACET)
	{
		// TODO
	}
	else if (bsdf.type == BsdfData::DISNEY)
	{
		// TODO
	}

	wo = wi;
	return ERROR_COLOR;
}

/**
 * Bsdf::sample, set wo and return eval / pdf * cosTheta
 */
static __forceinline__ __device__ float3 sampleBsdf(
		const BsdfData &bsdf, const float2 &uv, const float3 &wi,
		float3 &wo, float &pdf, unsigned int &seed)
{
	const float2 sample   = make_float2(rnd(seed), rnd(seed));
	const float  cosTheta = wi.z;

	if (bsdf.type == BsdfData::DIFFUSE)
	{
		wo = squareToCosineHemisphere(sample);

		if (bsdf.diffuse.albedoTex == 0)
			return bsdf.diffuse.albedo;
		else
			return make_float3(tex2D<float4>(bsdf.diffuse.albedoTex, uv.x, uv.y));
	}
	else if (bsdf.type == BsdfData::MIRROR)
	{
		wo = make_float3(-wi.x, -wi.y, wi.z);

		return make_float3(1.f);
	}
	else if (bsdf.type == BsdfData::DIELECTRIC)
	{
		// TODO
	}
	else if (bsdf.type == BsdfData::MICROFACET)
	{
		// TODO
	}
	else if (bsdf.type == BsdfData::DISNEY)
	{
		// TODO
	}

	wo = wi;
	return ERROR_COLOR;
}
