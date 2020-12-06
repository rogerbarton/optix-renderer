//
// Created by roger on 06/12/2020.
//

#include <nori/optix/OptixState.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>

#include "sutil/Exception.h"
#include "OptixSbtTypes.h"
#include "OptixState.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <map>

void OptixState::create()
{
	createContext();
	buildGases();
	buildIas();
	createPtxModules();
	createPipeline();
	createSbt();

	std::cout << "Optix state created.\n";
}

static void optixLogCallback(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "|" << std::setw(12) << tag << "]: " << message << "\n";
}

/**
 * Initializes cuda and optix
 * @param state Sets state.context
 */
void OptixState::createContext()
{
	CUDA_CHECK(cudaFree(nullptr));
	CUcontext                 cuCtx   = 0;

	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &optixLogCallback;
	options.logCallbackLevel    = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

	// Print device debug info
	int32_t deviceCount = 0;
	CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
	std::cout << "Total GPUs available: " << deviceCount << std::endl;

	for (int i = 0; i < deviceCount; ++i)
	{
		cudaDeviceProp prop = {};
		CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
		std::cout << "\t[" << i << "]: " << prop.name << " (" << prop.totalGlobalMem / 1024 / 1024 << "MB)"
		          << std::endl;
	}
}

/**
 * Creates program using the modules and creates the whole pipeline
 * @param state sets prog_groups and pipeline
 */
void OptixState::createPipeline()
{
	std::vector<OptixProgramGroup> program_groups;

	pipeline_compile_options = {};
	pipeline_compile_options.usesMotionBlur                   = false;
	pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipeline_compile_options.numPayloadValues                 = 3;
	pipeline_compile_options.numAttributeValues               = 6;
	pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
	pipeline_compile_options.pipelineLaunchParamsVariableName = "launchParams";

	// Prepare program groups
	createCameraProgram(program_groups);
	createHitProgram(program_groups);
	createVolumeProgram(program_groups);
	createMissProgram(program_groups);

	// Link program groups to pipeline
	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = maxTraceDepth;
	pipeline_link_options.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

	OPTIX_CHECK_LOG2(optixPipelineCreate(context,
	                                     &pipeline_compile_options,
	                                     &pipeline_link_options,
	                                     program_groups.data(),
	                                     static_cast<unsigned int>(program_groups.size()),
	                                     LOG,
	                                     &LOG_SIZE,
	                                     &pipeline));
}

void OptixState::createCameraProgram(std::vector<OptixProgramGroup> program_groups)
{
	OptixProgramGroup        cam_prog_group;
	OptixProgramGroupOptions cam_prog_group_options = {};
	OptixProgramGroupDesc    cam_prog_group_desc    = {};
	cam_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	cam_prog_group_desc.raygen.module            = camera_module;
	cam_prog_group_desc.raygen.entryFunctionName = "__raygen__perspective";

	OPTIX_CHECK_LOG2(optixProgramGroupCreate(
			context, &cam_prog_group_desc, 1, &cam_prog_group_options, LOG, &LOG_SIZE, &cam_prog_group));

	program_groups.push_back(cam_prog_group);
	raygen_prog_group = cam_prog_group;
}

void OptixState::createHitProgram(std::vector<OptixProgramGroup> program_groups)
{

}

void OptixState::createVolumeProgram(std::vector<OptixProgramGroup> program_groups)
{

	{
		OptixProgramGroup        radiance_prog_group;
		OptixProgramGroupOptions radiance_prog_group_options = {};
		OptixProgramGroupDesc    radiance_prog_group_desc    = {};
		radiance_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
				radiance_prog_group_desc.hitgroup.moduleIS = geometry_module;
		radiance_prog_group_desc.hitgroup.moduleCH         = shading_module;
		radiance_prog_group_desc.hitgroup.moduleAH         = nullptr;

		radiance_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
		radiance_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_radiance";
		radiance_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(context,
		                                         &radiance_prog_group_desc,
		                                         1,
		                                         &radiance_prog_group_options,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &radiance_prog_group));

		program_groups.push_back(radiance_prog_group);
		volume_prog_group[RAY_TYPE_RADIANCE] = radiance_prog_group;
	}

	{
		OptixProgramGroup        occlusion_prog_group;
		OptixProgramGroupOptions occlusion_prog_group_options = {};
		OptixProgramGroupDesc    occlusion_prog_group_desc    = {};
		occlusion_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
				occlusion_prog_group_desc.hitgroup.moduleIS    = geometry_module;
		occlusion_prog_group_desc.hitgroup.moduleCH            = nullptr;
		occlusion_prog_group_desc.hitgroup.moduleAH            = nullptr;
		occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
		occlusion_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

		occlusion_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
		occlusion_prog_group_desc.hitgroup.moduleCH            = shading_module;
		occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_occlusion";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(
				context, &occlusion_prog_group_desc, 1, &occlusion_prog_group_options, LOG, &LOG_SIZE,
				&occlusion_prog_group));

		program_groups.push_back(occlusion_prog_group);
		volume_prog_group[RAY_TYPE_OCCLUSION] = occlusion_prog_group;
	}
}

void OptixState::createMissProgram(std::vector<OptixProgramGroup> program_groups)
{
	{
		OptixProgramGroup        miss_prog_group;
		OptixProgramGroupOptions miss_prog_group_options = {};
		OptixProgramGroupDesc    miss_prog_group_desc    = {};

		miss_prog_group_desc.miss = {
				nullptr, // module
				nullptr // entryFunctionName
		};

		miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module            = shading_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__fogvolume_radiance";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(context,
		                                         &miss_prog_group_desc,
		                                         1,
		                                         &miss_prog_group_options,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &miss_prog_group[RAY_TYPE_RADIANCE]));
	}

	{
		OptixProgramGroup        miss_prog_group;
		OptixProgramGroupOptions miss_prog_group_options = {};
		OptixProgramGroupDesc    miss_prog_group_desc    = {};

		miss_prog_group_desc.miss = {
				nullptr, // module
				nullptr // entryFunctionName
		};

		miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module            = shading_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(context,
		                                         &miss_prog_group_desc,
		                                         1,
		                                         &miss_prog_group_options,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &miss_prog_group[RAY_TYPE_OCCLUSION]));
	}
}

void OptixState::createSbt()
{

	// Raygen program record
	{
		CUdeviceptr d_raygen_record;
		size_t      sizeof_raygen_record = sizeof(RayGenRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record), sizeof_raygen_record));

		sbt.raygenRecord = d_raygen_record;
	}

	// Miss program record
	{
		CUdeviceptr d_miss_record;
		size_t      sizeof_miss_record = sizeof(MissRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_record), sizeof_miss_record * RAY_TYPE_COUNT));

		MissRecord ms_sbt[RAY_TYPE_COUNT];
		for (int   i                   = 0; i < RAY_TYPE_COUNT; ++i)
		{
			optixSbtRecordPackHeader(miss_prog_group[i], &ms_sbt[i]);
			// data for miss program goes in here...
			//ms_sbt[i].data = {0.f, 0.f, 0.f};
		}

		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_miss_record),
		                      ms_sbt,
		                      sizeof_miss_record * RAY_TYPE_COUNT,
		                      cudaMemcpyHostToDevice));

		sbt.missRecordBase          = d_miss_record;
		sbt.missRecordCount         = RAY_TYPE_COUNT;
		sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof_miss_record);
	}

	// Hitgroup program record
	{
		const size_t                count_records = RAY_TYPE_COUNT;
		std::vector<HitGroupRecord> hitgroup_records(RAY_TYPE_COUNT);

		{
			int sbt_idx = 0;
			OPTIX_CHECK(
					optixSbtRecordPackHeader(volume_prog_group[RAY_TYPE_RADIANCE], &hitgroup_records[sbt_idx]));
			hitgroup_records[sbt_idx].data.geometry.volume = geometry;
			hitgroup_records[sbt_idx].data.shading.volume  = material;
			sbt_idx++;

			OPTIX_CHECK(
					optixSbtRecordPackHeader(volume_prog_group[RAY_TYPE_OCCLUSION], &hitgroup_records[sbt_idx]));
			hitgroup_records[sbt_idx].data.geometry.volume = geometry;
			hitgroup_records[sbt_idx].data.shading.volume  = material;
			sbt_idx++;
		}

		CUdeviceptr d_hitgroup_records;
		size_t      sizeof_hitgroup_record        = sizeof(HitGroupRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitgroup_records), sizeof_hitgroup_record * count_records));

		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitgroup_records),
		                      hitgroup_records.data(),
		                      sizeof_hitgroup_record * count_records,
		                      cudaMemcpyHostToDevice));

		sbt.hitgroupRecordBase          = d_hitgroup_records;
		sbt.hitgroupRecordCount         = count_records;
		sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof_hitgroup_record);
	}
}

void OptixState::render()
{
	if (ias_handle == 0)
	{
		std::cerr << "renderOptixState: state is not initialized.\n";
		return;
	}
	// TODO: map output buffer
	// TODO: update device copies, launch params etc

	OPTIX_CHECK(optixLaunch(pipeline,
	                        stream,
	                        reinterpret_cast<CUdeviceptr>(d_params),
	                        sizeof(LaunchParams),
	                        &sbt,
	                        width,
	                        height,
	                        1));

	CUDA_CHECK(cudaStreamSynchronize(stream));

	// TODO: unmap output buffer
}

OptixState::~OptixState()
{
	clear();
}

void OptixState::clear()
{
	throw std::exception("TODO: delete optixState.");
}
