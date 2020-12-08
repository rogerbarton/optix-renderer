//
// Created by roger on 08/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>
#include <vector_types.h>

struct TextureData
{
	enum Type
	{
		CONSTANT1f     = 0,
		CONSTANT3f     = 1,
		CHECKERBOARD1f = 2,
		CHECKERBOARD3f = 3,
		IMAGE1f        = 4,
		IMAGE3f        = 5,
		TYPE_COUNT
	};

	struct Constant1f
	{
		float value;
	};

	struct Constant3f
	{
		float3 value;
	};

	struct Checkerboard1f
	{
		float m_value1;
		float m_value2;

		float2 m_delta;
		float2 m_scale;
	};

	struct Checkerboard3f
	{
		float3 m_value1;
		float3 m_value2;

		float2 m_delta;
		float2 m_scale;
	};

	struct Image
	{
		cudaTextureObject_t texture;
	};

	Type type = CONSTANT1f;

	union
	{
		Constant1f constant1f;
		Constant3f constant3f;
		Checkerboard1f checkerboard1f;
		Checkerboard3f checkerboard3f;
		Image image;
	};
};