//
// Created by roger on 06/12/2020.
//

#pragma once

#include <optix.h>
#include "cuda/RayParams.h"

template<typename T>
struct Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T                                          data;
};

typedef Record<RaygenParams>   RaygenRecord;
typedef Record<MissParams>     MissRecord;
typedef Record<HitGroupParams> HitGroupRecord;
