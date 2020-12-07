//
// Created by roger on 06/12/2020.
//
#pragma once

#include <optix.h>

/**
 * Contains data used for the SBT
 */

#define NUM_PAYLOAD_VALUES 4
#define NUM_ATTRIBUTE_VALUES 2

enum RayType {
	RAY_TYPE_RADIANCE  = 0,
	RAY_TYPE_OCCLUSION = 1,
	RAY_TYPE_COUNT
};

struct RaygenParams {
};

struct MissParams {
};

struct HitGroupParams {
	float3 diffuseColor;
};
