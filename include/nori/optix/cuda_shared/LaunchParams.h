//
// Created by roger on 06/12/2020.
//
#pragma once

#include <optix_types.h>
#include <vector_types.h>
#include <texture_types.h>
#include "RaygenData.h"
#include "EmitterData.h"
#include "IntegratorData.h"

struct SceneConstantParams
{
	EmitterData* emitters;
	unsigned int emittersSize;
	int envmapIndex;
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

	RaygenData          camera;
	SceneConstantParams scene;

	IntegratorType integrator;

	OptixTraversableHandle sceneHandle;
};
