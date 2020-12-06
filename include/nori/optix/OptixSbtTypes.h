//
// Created by roger on 06/12/2020.
//

#pragma once

#include <optix.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cuda/MaterialData.h"

template<typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct Empty{};

//typedef Record<Camera>       RayGenRecord;
typedef Record<Empty>       RayGenRecord;
typedef Record<MissData> MissRecord;
typedef Record<HitGroupData> HitGroupRecord;
