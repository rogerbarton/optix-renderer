//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include <optix_types.h>

/**
 * The Optix params.
 * Note: this must be cuda compatible
 */
struct LaunchParams {
    unsigned int subframeIndex;
    float4* d_accumBuffer;
    uchar4* d_frameBuffer;
    unsigned int imageWidth;
    unsigned int imageHeight;

    unsigned int samplesPerLaunch;
    float3 eye;
    float3 U;
    float3 V;
    float3 W;

    float3 bgColor;

    OptixTraversableHandle sceneHandle;
};
#endif