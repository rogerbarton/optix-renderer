//
// Created by roger on 09/12/2020.
//

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
#include "sutil/LocalGeometry.h"

#include "shaders/bsdf.h"
#include "shaders/emitter.h"

#include "RadiancePrd.h"
#include "ShadowrayPrd.h"

extern "C" {
__constant__ LaunchParams launchParams;
}

extern "C" __global__ void __miss__shadowray()
{
	// no-op
	// ShadowrayPrd::setPrd(false);
}

extern "C" __global__ void __closesthit__shadowray()
{
	ShadowrayPrd::setPrd(true);
}

extern "C" __global__ void __miss__radiance()
{
	MissParams  *const rt_data = reinterpret_cast<MissParams *>( optixGetSbtDataPointer());
	RadiancePrd *const prd     = RadiancePrd::getPrd();

	// Evaluate the envmap
	if (launchParams.scene.envmapIndex >= 0)
	{
		const EmitterData &envmap = launchParams.scene.emitters[launchParams.scene.envmapIndex];
		if (envmap.environment.envmapTex)
		{
			float3 d = optixGetWorldRayDirection();
			float  u = acosf(d.z) * 0.5f / M_PIf;
			float  v = atan2f(d.y, d.x) / M_PIf;
			prd->Li += envmap.radiance * tex2D<float4>(envmap.environment.envmapTex, u, v);
		}
		else
			prd->Li += envmap.radiance * envmap.environment.envmapValue;
	}

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

	LocalGeometry lgeom = getLocalGeometry(sbtData->geometry, p);

	if (sbtData->emitter.type != EmitterData::NONE)
	{
		prd->Li += prd->throughput * evalEmitter(*sbtData, rayOrigin, p, lgeom.n);
	}

	if (sbtData->bsdf.type != BsdfData::NONE)
	{
		Frame  shFrame(lgeom.n);
		float3 wi      = shFrame.toLocal(-rayDir);
		float3 wo;
		float  pdf; // unused in mats
		prd->throughput *= sampleBsdf(sbtData->bsdf, lgeom.uv, wi, wo, pdf, prd->seed);
		prd->direction = shFrame.toWorld(wo);
	}

	prd->origin = p;
}
