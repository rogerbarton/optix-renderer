//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include "cuda_shared/LaunchParams.h"
#include "cuda_shared/RayParams.h"
#include <cuda.h>
#include <vector>

#include <nori/scene.h>

#include "sutil/CUDAOutputBuffer.h"

namespace nori
{
	class Shape;
}

struct GasHandle
{
	OptixTraversableHandle handle   = 0; /// The handle used for optix
	CUdeviceptr            d_buffer = 0; /// The allocated device buffer
};

/**
 * Stores all information about optix program.
 */
struct OptixState
{
	OptixDeviceContext m_context        = 0;
	CUstream           m_stream         = 0;
	bool               initializedOptix = false;

	LaunchParams *m_params   = nullptr;
	LaunchParams *m_d_params = nullptr;

	OptixTraversableHandle m_ias_handle          = {};
	CUdeviceptr            m_d_ias_output_buffer = 0;
	std::vector<GasHandle> m_gases;

	OptixModule             m_geometry_module                     = 0;
	OptixModule             m_camera_module                       = 0;
	OptixModule             m_shading_module                      = 0;
	OptixProgramGroup       m_raygen_prog_group                   = 0;
	OptixProgramGroup       m_miss_prog_group[RAY_TYPE_COUNT]     = {0, 0};
	OptixProgramGroup       m_hitgroup_prog_group[RAY_TYPE_COUNT] = {0, 0};
	OptixPipeline           m_pipeline                            = 0;
	OptixShaderBindingTable m_sbt                                 = {};

	OptixModuleCompileOptions   m_module_compile_options   = {};
	OptixPipelineCompileOptions m_pipeline_compile_options = {};
	OptixPipelineLinkOptions    m_pipeline_link_options    = {};

	const int maxTraceDepth = 2;

	// -- Interface
	void create();
	/**
	 * Call this before rendering. This will update the optix state and make it ready for rendering
	 * @return True if successful without errors and rendering can proceed
	 */
	bool preRender(nori::Scene& scene);
	/**
	 * Renders one subframe. Assumes that preRender has succeeded
	 * @param outputBuffer
	 */
	void renderSubframe(CUDAOutputBuffer<float4> &outputBuffer, uint32_t currentSample);
	void clear();
	~OptixState();

private:
	void createContext();
	void createCompileOptions();
	void buildGases(const std::vector<nori::Shape *> &shapes);
	void buildIas();
	void createPtxModules(bool specialize = true);
	void createPipeline();
	void createCameraProgram(std::vector<OptixProgramGroup> program_groups);
	void createHitProgram(std::vector<OptixProgramGroup> program_groups);
	void createVolumeProgram(std::vector<OptixProgramGroup> program_groups);
	void createMissProgram(std::vector<OptixProgramGroup> program_groups);
	void allocateSbt();
	void updateSbt(const std::vector<nori::Shape *> &shapes);
};

#endif