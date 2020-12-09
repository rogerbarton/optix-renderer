//
// Created by roger on 09/12/2020.
//
#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-reserved-identifier"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>
#include <vector_functions.h>
#include <vector_types.h>

#include "../cuda_shared/LaunchParams.h"
#include "../cuda_shared/RayParams.h"
#include "sutil/helpers.h"
#include "sutil/random.h"
#include "sutil/exception.h"

#include "shaders/bsdf.h"
#include "shaders/emitter.h"

#include "RadiancePrd.h"
#include "ShadowrayPrd.h"

extern "C" {
__constant__ LaunchParams launchParams;
}

extern "C" __global__ void __miss__occlusion()
{
	// no-op
	// ShadowrayPrd::setPrd(false);
}

extern "C" __global__ void __closesthit__occlusion()
{
	ShadowrayPrd::setPrd(true);
}

extern "C" __global__ void __miss__radiance()
{
	MissParams  *const rt_data = reinterpret_cast<MissParams *>( optixGetSbtDataPointer());
	RadiancePrd *const prd     = RadiancePrd::getPrd();

	// prd->Li += launchParams.envmap.sample; // TODO: sample envmap texture
	prd->terminated = true;
}

extern "C" __global__ void __closesthit__radiance()
{
	HitGroupParams *sbtData = reinterpret_cast<HitGroupParams *>( optixGetSbtDataPointer());
	RadiancePrd    *prd     = RadiancePrd::getPrd();

	const OptixPrimitiveType type      = optixGetPrimitiveType();
	const uint32_t           primIdx   = optixGetPrimitiveIndex();
	const float3             rayOrigin = optixGetWorldRayOrigin();
	const float3             rayDir    = optixGetWorldRayDirection();
	const float3             p         = rayOrigin + optixGetRayTime() * rayDir;;
	const uint32_t vertIdxOffset = primIdx * 3;

	float3 normal =;
	float2 uv     =;

	if (sbtData->emitter.type != EmitterData::NONE)
	{
		prd->Li += prd->throughput * evalEmitter(*sbtData, rayOrigin, p, normal);
	}

	if (sbtData->bsdf.type != BsdfData::NONE)
	{
		Frame  shFrame(normal);
		float3 wi      = shFrame.toLocal(-rayDir);
		float3 wo;
		float  pdf; // unused in mats
		prd->throughput *= sampleBsdf(sbtData->bsdf, uv, wi, wo, pdf, prd->seed);
		prd->direction = shFrame.toWorld(wo);
	}

	prd->origin = p;
}

#pragma clang diagnostic pop