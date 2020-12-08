//
// Created by roger on 08/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>
#include <vector_types.h>

struct MaterialData
{
	enum Type
	{
		DIFFUSE    = 0,
		MIRROR     = 1,
		DIELECTRIC = 2,
		MICROFACET = 3,
		DISNEY     = 4,
		TYPE_COUNT
	};

	struct Diffuse
	{
		float3 albedo = {0.5f, 0.5f, 0.5f};
	};

	struct Mirror
	{

	};

	struct Pbr
	{
		float4 base_color = {1.0f, 1.0f, 1.0f, 1.0f};
		float  metallic   = 1.0f;
		float  roughness  = 1.0f;

		cudaTextureObject_t base_color_tex         = 0;
		cudaTextureObject_t metallic_roughness_tex = 0;
		cudaTextureObject_t normal_tex             = 0;
	};

	Type type;

	union
	{
		Diffuse diffuse;
		Pbr     pbr;
	};
};