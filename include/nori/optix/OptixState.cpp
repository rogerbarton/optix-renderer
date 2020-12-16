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
	CUDA_CHECK(cudaStreamCreate(&m_stream));

	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &optixLogCallback;
#ifdef NORI_OPTIX_RELEASE
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
	if (initializedState)
		return;

	m_module_compile_options = {};
	m_module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NORI_OPTIX_RELEASE
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
#ifdef NORI_OPTIX_RELEASE
	m_pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
#else
	m_pipeline_compile_options.exceptionFlags =
			OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
			OPTIX_EXCEPTION_FLAG_USER;
#endif
	m_pipeline_compile_options.pipelineLaunchParamsVariableName = "launchParams";

	m_pipeline_link_options = {};
	m_pipeline_link_options.maxTraceDepth = maxTraceDepth;
#ifdef NORI_OPTIX_RELEASE
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
	createRaygenProgram(program_groups);
	createMissProgram(program_groups);
	createHitMeshProgram(program_groups);
	createHitSphereProgram(program_groups);
	createHitVolumeProgram(program_groups);

	// Link program groups to pipeline
	OPTIX_CHECK_LOG2(optixPipelineCreate(m_context,
	                                     &m_pipeline_compile_options,
	                                     &m_pipeline_link_options,
	                                     program_groups.data(), static_cast<unsigned int>(program_groups.size()),
	                                     LOG, &LOG_SIZE,
	                                     &m_pipeline));
}

void OptixState::createRaygenProgram(std::vector<OptixProgramGroup> &program_groups)
{
	OptixProgramGroupOptions rayGenProgGroupOptions = {};
	OptixProgramGroupDesc    raygenProgGroupDesc    = {};
	raygenProgGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	raygenProgGroupDesc.raygen.module            = m_raygen_module;
	raygenProgGroupDesc.raygen.entryFunctionName = "__raygen__perspective";

	OPTIX_CHECK_LOG2(optixProgramGroupCreate(
			m_context, &raygenProgGroupDesc, 1, &rayGenProgGroupOptions, LOG, &LOG_SIZE, &m_raygen_prog_group));

	program_groups.push_back(m_raygen_prog_group);
}

void OptixState::createMissProgram(std::vector<OptixProgramGroup> &program_groups)
{
	{
		OptixProgramGroupOptions missProgGroupOptions = {};
		OptixProgramGroupDesc    missProgGroupDesc    = {};

		missProgGroupDesc.miss = {
				nullptr, // module
				nullptr // entryFunctionName
		};

		missProgGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		missProgGroupDesc.miss.module            = m_shading_module;
		missProgGroupDesc.miss.entryFunctionName = "__miss__radiance";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &missProgGroupDesc,
		                                         1,
		                                         &missProgGroupOptions,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &m_miss_prog_group[RAY_TYPE_RADIANCE]));
	}

	{
		OptixProgramGroupOptions missProgGroupOptions = {};
		OptixProgramGroupDesc    missProgGroupDesc    = {};

		missProgGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		missProgGroupDesc.miss.module            = m_shading_module;
		missProgGroupDesc.miss.entryFunctionName = "__miss__shadowray";

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &missProgGroupDesc,
		                                         1,
		                                         &missProgGroupOptions,
		                                         LOG,
		                                         &LOG_SIZE,
		                                         &m_miss_prog_group[RAY_TYPE_SHADOWRAY]));
	}
}

void OptixState::createHitMeshProgram(std::vector<OptixProgramGroup> &program_groups)
{
	/**
	 * Hit programs for Meshes
	 * No intersection or anyhit programs
	 */

	OptixProgramGroupOptions progGroupOptions = {};
	{
		OptixProgramGroupDesc radianceProgGroupDesc = {};
		radianceProgGroupDesc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		radianceProgGroupDesc.hitgroup.moduleIS = nullptr;
		radianceProgGroupDesc.hitgroup.moduleCH = m_shading_module;
		radianceProgGroupDesc.hitgroup.moduleAH = nullptr;

		radianceProgGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
		radianceProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		radianceProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &radianceProgGroupDesc, 1,
		                                         &progGroupOptions,
		                                         LOG, &LOG_SIZE,
		                                         &m_hitgroup_mesh_prog_group[RAY_TYPE_RADIANCE]));

		program_groups.push_back(m_hitgroup_mesh_prog_group[RAY_TYPE_RADIANCE]);
	}

	{
		OptixProgramGroupDesc occlusionProgGroupDesc = {};
		occlusionProgGroupDesc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		occlusionProgGroupDesc.hitgroup.moduleIS = nullptr;
		occlusionProgGroupDesc.hitgroup.moduleCH = m_shading_module;
		occlusionProgGroupDesc.hitgroup.moduleAH = nullptr;

		occlusionProgGroupDesc.hitgroup.entryFunctionNameIS = nullptr;
		occlusionProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadowray";
		occlusionProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &occlusionProgGroupDesc, 1,
		                                         &progGroupOptions,
		                                         LOG, &LOG_SIZE,
		                                         &m_hitgroup_mesh_prog_group[RAY_TYPE_SHADOWRAY]));

		program_groups.push_back(m_hitgroup_mesh_prog_group[RAY_TYPE_SHADOWRAY]);
	}
}

void OptixState::createHitSphereProgram(std::vector<OptixProgramGroup> &program_groups)
{
	/**
	 * Hit programs for everything except volumes
	 * No anyhit program
	 */

	OptixProgramGroupOptions progGroupOptions = {};
	{
		OptixProgramGroupDesc radianceProgGroupDesc = {};
		radianceProgGroupDesc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		radianceProgGroupDesc.hitgroup.moduleIS = m_geometry_sphere_module;
		radianceProgGroupDesc.hitgroup.moduleCH = m_shading_module;
		radianceProgGroupDesc.hitgroup.moduleAH = nullptr;

		radianceProgGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
		radianceProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		radianceProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &radianceProgGroupDesc, 1,
		                                         &progGroupOptions,
		                                         LOG, &LOG_SIZE,
		                                         &m_hitgroup_sphere_prog_group[RAY_TYPE_RADIANCE]));

		program_groups.push_back(m_hitgroup_sphere_prog_group[RAY_TYPE_RADIANCE]);
	}

	{
		OptixProgramGroupDesc occlusionProgGroupDesc = {};
		occlusionProgGroupDesc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		occlusionProgGroupDesc.hitgroup.moduleIS = m_geometry_sphere_module;
		occlusionProgGroupDesc.hitgroup.moduleCH = m_shading_module;
		occlusionProgGroupDesc.hitgroup.moduleAH = nullptr;

		occlusionProgGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
		occlusionProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadowray";
		occlusionProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &occlusionProgGroupDesc, 1,
		                                         &progGroupOptions,
		                                         LOG, &LOG_SIZE,
		                                         &m_hitgroup_sphere_prog_group[RAY_TYPE_SHADOWRAY]));

		program_groups.push_back(m_hitgroup_sphere_prog_group[RAY_TYPE_SHADOWRAY]);
	}
}

void OptixState::createHitVolumeProgram(std::vector<OptixProgramGroup> &program_groups)
{
	// TODO: implement these
	return;

	OptixProgramGroupOptions progGroupOptions = {};
	{
		OptixProgramGroupDesc radianceProgGroupDesc = {};
		radianceProgGroupDesc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		radianceProgGroupDesc.hitgroup.moduleIS = m_geometry_nvdb_module;
		radianceProgGroupDesc.hitgroup.moduleCH = m_shading_module;
		radianceProgGroupDesc.hitgroup.moduleAH = nullptr;

		radianceProgGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
		radianceProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_radiance"; // TODO
		radianceProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &radianceProgGroupDesc, 1,
		                                         &progGroupOptions,
		                                         LOG, &LOG_SIZE,
		                                         &m_hitgroup_nvdb_prog_group[RAY_TYPE_RADIANCE]));

		program_groups.push_back(m_hitgroup_nvdb_prog_group[RAY_TYPE_RADIANCE]);
	}

	{
		OptixProgramGroupOptions occlusionProgGroupOptions = {};
		OptixProgramGroupDesc    occlusionProgGroupDesc    = {};
		occlusionProgGroupDesc.kind              = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		occlusionProgGroupDesc.hitgroup.moduleIS = m_geometry_nvdb_module;
		occlusionProgGroupDesc.hitgroup.moduleCH = m_shading_module;
		occlusionProgGroupDesc.hitgroup.moduleAH = nullptr;

		occlusionProgGroupDesc.hitgroup.entryFunctionNameIS = "__intersection__nanovdb_fogvolume";
		occlusionProgGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__nanovdb_fogvolume_shadowray"; // TODO
		occlusionProgGroupDesc.hitgroup.entryFunctionNameAH = nullptr;

		OPTIX_CHECK_LOG2(optixProgramGroupCreate(m_context,
		                                         &occlusionProgGroupDesc, 1,
		                                         &progGroupOptions,
		                                         LOG, &LOG_SIZE,
		                                         &m_hitgroup_nvdb_prog_group[RAY_TYPE_SHADOWRAY]));

		program_groups.push_back(m_hitgroup_nvdb_prog_group[RAY_TYPE_SHADOWRAY]);
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

	CUDA_CHECK(cudaStreamDestroy(m_stream));

}
void OptixState::clearPipeline()
{
	if (m_pipeline)
		OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
	if (m_raygen_prog_group)
		OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_prog_group));

	for (int i = 0; i < RAY_TYPE_COUNT; ++i)
	{
		if (m_miss_prog_group[i])
			OPTIX_CHECK(optixProgramGroupDestroy(m_miss_prog_group[i]));
		if (m_hitgroup_mesh_prog_group[i])
			OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_mesh_prog_group[i]));
		if (m_hitgroup_sphere_prog_group[i])
			OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_sphere_prog_group[i]));
		if (m_hitgroup_nvdb_prog_group[i])
			OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_nvdb_prog_group[i]));
	}

	if (m_raygen_module)
		OPTIX_CHECK(optixModuleDestroy(m_raygen_module));
	if (m_geometry_sphere_module)
		OPTIX_CHECK(optixModuleDestroy(m_geometry_sphere_module));
	if (m_geometry_nvdb_module)
		OPTIX_CHECK(optixModuleDestroy(m_geometry_nvdb_module));
	if (m_shading_module)
		OPTIX_CHECK(optixModuleDestroy(m_shading_module));

	m_raygen_prog_group = 0;
	m_pipeline          = 0;
	m_raygen_prog_group = 0;

	for (int i = 0; i < RAY_TYPE_COUNT; ++i)
	{
		m_miss_prog_group[i]            = 0;
		m_miss_prog_group[i]            = 0;
		m_hitgroup_mesh_prog_group[i]   = 0;
		m_hitgroup_sphere_prog_group[i] = 0;
		m_hitgroup_nvdb_prog_group[i]   = 0;
	}
	m_raygen_module          = 0;
	m_geometry_sphere_module = 0;
	m_geometry_nvdb_module   = 0;
	m_shading_module         = 0;
}


void OptixState::renderSubframe(const uint32_t currentSample,
                                float4 *outputBuffer, float4 *outputBufferAlbedo, float4 *outputBufferNormal)
{
	// Update params and copy to device
	m_params.sampleIndex          = currentSample;
	m_params.d_outputBuffer       = outputBuffer;
	m_params.d_outputBufferAlbedo = outputBufferAlbedo;
	m_params.d_outputBufferNormal = outputBufferNormal;

	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(m_d_params), &m_params, sizeof(LaunchParams),
	                           cudaMemcpyHostToDevice, m_stream));

	// std::cout << "optixLaunch " << currentSample << std::endl;
	OPTIX_CHECK(optixLaunch(m_pipeline,
	                        m_stream,
	                        reinterpret_cast<CUdeviceptr>(m_d_params),
	                        sizeof(LaunchParams),
	                        &m_sbt,
	                        m_params.imageWidth,
	                        m_params.imageHeight,
	                        maxTraceDepth));

	CUDA_CHECK(cudaStreamSynchronize(m_stream));
	CUDA_SYNC_CHECK();
	// std::cout << "optixLaunch done." << std::endl;
}
