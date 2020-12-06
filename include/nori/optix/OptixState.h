//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include "cuda/LaunchParams.h"
#include "cuda/MaterialData.h"
#include <cuda.h>

/**
 * Stores all information about our app
 */
struct OptixState
{
	OptixDeviceContext          context = 0;
	CUstream                    stream = 0;

	LaunchParams*                     params = nullptr;
	LaunchParams*                     d_params = nullptr;

	OptixTraversableHandle      ias_handle = {};
	CUdeviceptr                 d_ias_output_buffer = {};

	OptixModule                 geometry_module = 0;
	OptixModule                 camera_module = 0;
	OptixModule                 shading_module = 0;
	OptixProgramGroup           raygen_prog_group = 0;
	OptixProgramGroup           miss_prog_group[RAY_TYPE_COUNT] = {0, 0};
	OptixProgramGroup           volume_prog_group[RAY_TYPE_COUNT] = {0, 0};
	OptixPipeline               pipeline = 0;
	OptixPipelineCompileOptions pipeline_compile_options = {};
	OptixShaderBindingTable     sbt = {};

    const int maxTraceDepth = 10;
};
#endif