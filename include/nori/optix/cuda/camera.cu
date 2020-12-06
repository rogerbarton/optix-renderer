//
// Created by roger on 06/12/2020.
//

#include <optix_device.h>
#include "LaunchParams.h"
#include "sutil/helpers.h"
#include <nanovdb/util/Ray.h>

extern "C" {
    __constant__ LaunchParams launchParams;
}


extern "C" __global__ void __raygen__perspective()
{
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using CoordT = nanovdb::Coord;
    using RayT = nanovdb::Ray<RealT>;

    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();
    int            ix = idx.x;
    int            iy = idx.y;
    const uint32_t offset = launchParams.width * idx.y + idx.x;
    const auto&    sceneParams = launchParams.sceneConstants;

    float3 color = {0, 0, 0};

    for (int sampleIndex = 0; sampleIndex < sceneParams.samplesPerPixel; ++sampleIndex) {
        uint32_t pixelSeed = render::hash((sampleIndex + (launchParams.numAccumulations + 1) * sceneParams.samplesPerPixel)) ^ render::hash(ix, iy);

        RayT wRay = render::getRayFromPixelCoord(ix, iy, launchParams.width, launchParams.height, launchParams.numAccumulations, sceneParams.samplesPerPixel, pixelSeed, sceneParams);

        float3 result;
        optixTrace(
                launchParams.handle,
                make_float3(wRay.eye()),
                make_float3(wRay.dir()),
                launchParams.sceneEpsilon,
                1e16f,
                0.0f,
                OptixVisibilityMask(1),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                RAY_TYPE_RADIANCE,
                RAY_TYPE_COUNT,
                RAY_TYPE_RADIANCE,
                float3_as_args(result));

        color += result;
    }

    color /= (float)sceneParams.samplesPerPixel;

    if (launchParams.numAccumulations > 1) {
        float3 prevPixel = make_float3(launchParams.imgBuffer[offset]);

        color = prevPixel + (color - prevPixel) * (1.0f / launchParams.numAccumulations);
    }

    launchParams.imgBuffer[offset] = make_float4(color, 1.f);
}

__forceinline__ float3 Li(){

}