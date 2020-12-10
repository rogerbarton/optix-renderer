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

#include "OptixState.h"
#include "OptixSbtTypes.h"
#include "cuda_shared/GeometryData.h"

#include <nori/mesh.h>
#include <nori/sphere.h>

#include "sutil/Exception.h"
#include "sutil/host_vec_math.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <map>

void OptixState::create()
{
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
	if (initializedOptix)
		return;
	initializedOptix                  = true;

	CUDA_CHECK(cudaFree(nullptr));
	CUcontext                 cuCtx   = 0;

	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &optixLogCallback;
#ifdef NDEBUG
	options.logCallbackLevel    = 2;
#else
	options.logCallbackLevel = 4;
#endif
#if OPTIX_VERSION >= 70200
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));

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
 * Sets compile and link options:
 * module_compile_options, pipeline_compile_options, pipeline_link_options
 */
void OptixState::createCompileOptions()
{
	if (initializedOptix)
		return;

	m_module_compile_options = {};
	m_module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
	m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
	m_module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	m_module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

	m_pipeline_compile_options = {};
	m_pipeline_compile_options.usesMotionBlur        = false;
	m_pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
	m_pipeline_compile_options.numPayloadValues      = NUM_PAYLOAD_VALUES;
	m_pipeline_compile_options.numAttributeValues    = NUM_ATTRIBUTE_VALUES;
#ifdef NDEBUG
	m_pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
#else
	m_pipeline_compile_options.exceptionFlags =
			OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
			OPTIX_EXCEPTION_FLAG_USER;
#endif
	m_pipeline_compile_options.pipelineLaunchParamsVariableName = "launchParams";

	m_pipeline_link_options = {};
	m_pipeline_link_options.maxTraceDepth = maxTraceDepth;
#ifdef NDEBUG
	m_pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
	m_pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
}

/**
 * Creates program using the modules and creates the whole pipeline
 * @param state sets prog_groups and pipeline
 */
void OptixState::createPipeline()
{
	std::vector<OptixProgramGroup> program_groups;

	// Prepare program groups
	createCameraProgram(program_groups);
	createHitProgram(program_groups);
	createVolumeProgram(program_groups);
	createMissProgram(program_groups);

	// Link program groups to pipeline
	OPTIX_CHECK_LOG2(optixPipelineCreate(m_context,
	                                     &m_pipeline_compile_options,
	                                     &m_pipeline_link_options,
	                                     program_groups.data(), static_cast<unsigned int>(program_groups.size()),
	                                     LOG, &LOG_SIZE,
	                                     &m_pipeline));
}

void OptixState::createCameraProgram(std::vector<OptixProgramGroup> program_groups)
{
	OptixProgramGroupOptions cam_prog_group_options = {};
	OptixProgramGroupDesc    cam_prog_group_desc    = {};
	cam_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	cam_prog_group_desc.raygen.module            = m_camera_module;
	cam_prog_group_desc.raygen.entryFunctionName = "__raygen__perspective";

	OPTIX_CHECK_LOG2(optixProgramGroupCreate(
			m_context, &cam_prog_group_desc, 1, &cam_prog_group_options, LOG, &LOG_SIZE, &m_raygen_prog_group));

	program_groups.push_back(m_raygen_prog_group);
}

void OptixState::createHitProgram(std::vector<OptixProgramGroup> program_groups)
{

}

void OptixState::createVolumeProgram(std::vector<OptixProgramGroup> program_groups)
{
	{
		OptixProgramGroupOptions radiance_prog_group_options = {};
		OptixProgramGroupDesc    radiance_prog_group_desc    = {};
		radiance_prog_group_desc.kind                      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
				radiance_prog_group_desc.hitgroup.moduleIS = m_geometry_module;
		radiance_prog_group_desc.hitgroup.moduleCH         = m_shading_module;
		radiance_prog_group_desc.hitgroup.moduleAH         = nullptr;

		radiance_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
		radiance_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_radiance";
		radiance_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &radiance_prog_group_desc,
		                                         1,
		                                         &radiance_prog_group_options,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &m_hitgroup_prog_group[RAY_TYPE_RADIANCE]));

		program_groups.push_back(m_hitgroup_prog_group[RAY_TYPE_RADIANCE]);
	}

	{
		OptixProgramGroupOptions occlusion_prog_group_options = {};
		OptixProgramGroupDesc    occlusion_prog_group_desc    = {};
		occlusion_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
				occlusion_prog_group_desc.hitgroup.moduleIS    = m_geometry_module;
		occlusion_prog_group_desc.hitgroup.moduleCH            = nullptr;
		occlusion_prog_group_desc.hitgroup.moduleAH            = nullptr;
		occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
		occlusion_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;

		occlusion_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
		occlusion_prog_group_desc.hitgroup.moduleCH            = m_shading_module;
		occlusion_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_occlusion";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(
				m_context, &occlusion_prog_group_desc, 1, &occlusion_prog_group_options, LOG, &LOG_SIZE,
				&m_hitgroup_prog_group[RAY_TYPE_SHADOWRAY]));

		program_groups.push_back(m_hitgroup_prog_group[RAY_TYPE_SHADOWRAY]);
	}
}

void OptixState::createMissProgram(std::vector<OptixProgramGroup> program_groups)
{
	{
		OptixProgramGroupOptions miss_prog_group_options = {};
		OptixProgramGroupDesc    miss_prog_group_desc    = {};

		miss_prog_group_desc.miss = {
				nullptr, // module
				nullptr // entryFunctionName
		};

		miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module            = m_shading_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__fogvolume_radiance";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &miss_prog_group_desc,
		                                         1,
		                                         &miss_prog_group_options,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &m_miss_prog_group[RAY_TYPE_RADIANCE]));
	}

	{
		OptixProgramGroupOptions miss_prog_group_options = {};
		OptixProgramGroupDesc    miss_prog_group_desc    = {};

		miss_prog_group_desc.miss = {
				nullptr, // module
				nullptr // entryFunctionName
		};

		miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module            = m_shading_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &miss_prog_group_desc,
		                                         1,
		                                         &miss_prog_group_options,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &m_miss_prog_group[RAY_TYPE_SHADOWRAY]));
	}
}

void OptixState::updateSbt(const std::vector<nori::Shape *> &shapes)
{
	// Create the records on the host and copy to the already allocated device sbt
	// Empty records still need to be copied for the optix sbt record header

	// -- Raygen = empty record data
	{
		RaygenRecord raygenRecord = {};
		OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &raygenRecord));

		if (!initializedState)
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_sbt.raygenRecord), sizeof(RaygenRecord)));

		CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void *>(m_sbt.raygenRecord),
				&raygenRecord,
				sizeof(RaygenRecord),
				cudaMemcpyHostToDevice));
	}


	// -- Miss = empty record data
	{
		MissRecord missRecords[2];
		OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_prog_group[0], &missRecords[0]));
		OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_prog_group[1], &missRecords[1]));

		if (!initializedState)
		{
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_sbt.missRecordBase),
			                      sizeof(MissRecord) * RAY_TYPE_COUNT));
			m_sbt.missRecordCount         = RAY_TYPE_COUNT;
			m_sbt.missRecordStrideInBytes = static_cast<uint32_t>(sizeof(MissRecord));
		}

		CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void *>(m_sbt.missRecordBase),
				missRecords,
				sizeof(MissRecord) * RAY_TYPE_COUNT,
				cudaMemcpyHostToDevice));
	}


	// Hitgroup program record
	{
		std::vector<HitGroupRecord> hitgroupRecords;
		hitgroupRecords.reserve(shapes.size() * RAY_TYPE_COUNT);
		for (nori::Shape *shape : shapes)
		{
			shape->getOptixHitgroupRecords(*this, hitgroupRecords);
		}

		if (!initializedState)
		{
			CUDA_CHECK(cudaMalloc(
					reinterpret_cast<void **>( &m_sbt.hitgroupRecordBase ),
					hitgroupRecords.size() * sizeof(HitGroupRecord)));
			m_sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(sizeof(HitGroupRecord));
			m_sbt.hitgroupRecordCount         = static_cast<uint32_t>(hitgroupRecords.size());
		}

		CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void *>( m_sbt.hitgroupRecordBase ),
				hitgroupRecords.data(),
				hitgroupRecords.size() * sizeof(HitGroupRecord),
				cudaMemcpyHostToDevice));
	}
}

OptixState::~OptixState()
{
	clear();
}

void OptixState::clear()
{
	clearPipeline();

	OPTIX_CHECK(optixDeviceContextDestroy(m_context));

	CUDA_CHECK(cudaFree(reinterpret_cast<void *>( m_sbt.raygenRecord )));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>( m_sbt.missRecordBase )));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>( m_sbt.hitgroupRecordBase )));

	for (auto &gas : m_gases)
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(gas.d_buffer)));
	m_gases.clear();
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>( m_d_ias_output_buffer)));

	CUDA_CHECK(cudaFree(reinterpret_cast<void *>( m_d_params )));

}
void OptixState::clearPipeline()
{
	if (m_pipeline)
		OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
	if (m_raygen_prog_group)
		OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_prog_group));
	if (m_miss_prog_group[RAY_TYPE_RADIANCE])
		OPTIX_CHECK(optixProgramGroupDestroy(m_miss_prog_group[RAY_TYPE_RADIANCE]));
	if (m_miss_prog_group[RAY_TYPE_SHADOWRAY])
		OPTIX_CHECK(optixProgramGroupDestroy(m_miss_prog_group[RAY_TYPE_SHADOWRAY]));
	if (m_hitgroup_prog_group[RAY_TYPE_RADIANCE])
		OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_prog_group[RAY_TYPE_RADIANCE]));
	if (m_hitgroup_prog_group[RAY_TYPE_SHADOWRAY])
		OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_prog_group[RAY_TYPE_SHADOWRAY]));
	if (m_camera_module)
		OPTIX_CHECK(optixModuleDestroy(m_camera_module));
	if (m_geometry_module)
		OPTIX_CHECK(optixModuleDestroy(m_geometry_module));
	if (m_shading_module)
		OPTIX_CHECK(optixModuleDestroy(m_shading_module));

	m_raygen_prog_group = 0;
	m_pipeline          = 0;
	m_raygen_prog_group = 0;
	m_miss_prog_group[RAY_TYPE_RADIANCE]      = 0;
	m_miss_prog_group[RAY_TYPE_SHADOWRAY]     = 0;
	m_hitgroup_prog_group[RAY_TYPE_RADIANCE]  = 0;
	m_hitgroup_prog_group[RAY_TYPE_SHADOWRAY] = 0;
	m_camera_module   = 0;
	m_geometry_module = 0;
	m_shading_module  = 0;
}


void OptixState::renderSubframe(CUDAOutputBuffer<float4> &outputBuffer, uint32_t currentSample)
{
	// Update params and copy to device
	m_params.sampleIndex   = currentSample;
	m_params.d_imageBuffer = outputBuffer.map();

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(&m_params), &m_d_params, sizeof(LaunchParams),
	                           cudaMemcpyHostToDevice, m_stream));

	OPTIX_CHECK(optixLaunch(m_pipeline,
	                        m_stream,
	                        reinterpret_cast<CUdeviceptr>(m_d_params),
	                        sizeof(LaunchParams),
	                        &m_sbt,
	                        m_params.imageWidth,
	                        m_params.imageHeight,
	                        1));
	outputBuffer.unmap();

	CUDA_CHECK(cudaStreamSynchronize(m_stream));
	CUDA_SYNC_CHECK();

}