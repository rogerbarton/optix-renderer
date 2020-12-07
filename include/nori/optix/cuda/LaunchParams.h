//
// Created by roger on 06/12/2020.
//
#pragma once

#include <optix.h>
#include <vector_types.h>

#define Epsilon 1e-4f

struct RaygenConstantParams
{
	float3 eye;
	float3 U;
	float3 V;
	float3 W;
};

struct SceneConstantParams
{
	float3 bgColor;
};

/**
 * The Optix params.
 * Note: this must be cuda compatible
 */
struct LaunchParams
{
	unsigned int sampleIndex;
	unsigned int samplesPerLaunch;
	float4 *d_imageBuffer;
	unsigned int imageWidth;
	unsigned int imageHeight;

	RaygenConstantParams camera;
	SceneConstantParams  scene;

	OptixTraversableHandle sceneHandle;
};
