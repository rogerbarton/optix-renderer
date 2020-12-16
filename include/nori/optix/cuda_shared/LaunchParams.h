//
// Created by roger on 06/12/2020.
//
#pragma once

#include <optix_types.h>
#include <vector_types.h>
#include "RaygenData.h"

struct SceneConstantParams
{
	float3 bgColor;
};

enum IntegratorType : int
{
	INTEGRATOR_TYPE_PATH_MATS = 0,
	INTEGRATOR_TYPE_PATH_MIS  = 1,
	INTEGRATOR_TYPE_DIRECT    = 2,
	INTEGRATOR_TYPE_SIZE
};

/**
 * The Optix params.
 * Note: this must be cuda compatible
 */
struct LaunchParams
{
	unsigned int sampleIndex;
	unsigned int samplesPerLaunch;
	float4       *d_outputBuffer;
	float4       *d_outputBufferAlbedo;
	float4       *d_outputBufferNormal;
	unsigned int imageWidth;
	unsigned int imageHeight;

	RaygenData camera;
	SceneConstantParams  scene;

	IntegratorType integrator;

	OptixTraversableHandle sceneHandle;
};
