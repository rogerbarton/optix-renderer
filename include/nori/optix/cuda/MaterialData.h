//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include <optix.h>
#include "sutil/GeometryData.h"

#define NUM_PAYLOAD_VALUES 4
#define NUM_ATTRIBUTE_VALUES 2

enum RayType {
	RAY_TYPE_RADIANCE  = 0,
	RAY_TYPE_OCCLUSION = 1,
	RAY_TYPE_COUNT
};

struct RayGenData {
};

struct MissData {
};

struct HitGroupData {
	GeometryData geometryData;
	float3 diffuseColor;
};

#endif
