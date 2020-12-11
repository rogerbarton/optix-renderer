//
// Created by roger on 08/12/2020.
//

#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>
#include <vector_types.h>

struct EmitterData
{
	enum Type
	{
		NONE   = 0,
		POINT  = 1,
		SPOT   = 2,
		AREA   = 3,
		VOLUME = 4,
		DIRECTIONAL = 5,
		TYPE_COUNT
	};

	struct Point
	{
		float3 position;
	};

	struct Spot
	{
		float3 position;
		float3 direction;
		float  totalWidthAngle; /// TODO Note: angles in radians
		float  falloffStart;
	};

	struct Area
	{
	};

	struct Volume
	{
		float3 invVolume; // TODO: set this, so we can use pdf directly
	};

	struct Directional
	{
		float3 direction;
		float3 position;
		float angle;
	};

	Type   type = NONE;
	float3 radiance;

	union
	{
		Point point;
		Spot spot;
		Area area;
		Volume volume;
		Directional directional;
	};
};