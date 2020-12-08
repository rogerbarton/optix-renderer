//
// Created by roger on 08/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>
#include <vector_types.h>

struct MaterialData
{
	/**
	 * Note: ids need to match BSDF::getBsdfId()
	 */
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
		float3              albedo    = {0.5f, 0.5f, 0.5f};
		cudaTextureObject_t albedoTex = 0;
	};

	struct Mirror
	{
	};

	struct Dielectric
	{
		float intIOR, extIOR;
	};

	struct Microfacet
	{
		float  m_alpha;
		float  m_intIOR, m_extIOR;
		float  m_ks;
		float3 m_kd;
	};

	struct Disney
	{
		float3 albedo;
		float  metallic;
		float  subsurface;
		float  specular;
		float  roughness;
		float  specularTint;
		float  anisotropic;
		float  sheen;
		float  sheenTint;
		float  clearcoat;
		float  clearcoatGloss;

		cudaTextureObject_t albedoTex = 0;
	};

	Type                type;
	cudaTextureObject_t normalTex = 0;

	union
	{
		Diffuse    diffuse;
		Dielectric dielectric;
		Mirror     mirror;
		Microfacet microfacet;
		Disney     disney;
	};
};