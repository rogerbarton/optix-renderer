//
// Created by roger on 06/12/2020.
//
#pragma once

#include <optix.h>
#include "GeometryData.h"
#include "BsdfData.h"
#include "MediumData.h"
#include "EmitterData.h"

/**
 * Contains data used for the SBT
 */

// Payload passed from hit programs to back to the optixTrace caller
#define NUM_PAYLOAD_VALUES 2
// Attributes passed from intersect programs to hit programs
#define NUM_ATTRIBUTE_VALUES 4

enum RayType
{
	RAY_TYPE_RADIANCE  = 0,
	RAY_TYPE_SHADOWRAY = 1,
	RAY_TYPE_COUNT
};

struct RaygenParams
{
};

struct MissParams
{
};

struct HitGroupParams
{
	GeometryData geometry;
	BsdfData     bsdf;
	MediumData   medium;
	EmitterData  emitter;
};
