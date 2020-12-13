//
// Created by roger on 06/12/2020.
//

#pragma once
#ifdef NORI_USE_OPTIX

#include "cuda_shared/LaunchParams.h"
#include "cuda_shared/RayParams.h"
#include <cuda.h>
#include <vector>

namespace nori
{
	class Scene;
}

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
	bool               initializedState = false;

	LaunchParams m_params    = {};
	LaunchParams *m_d_params = nullptr;

	OptixTraversableHandle m_ias_handle          = {};
	CUdeviceptr            m_d_ias_output_buffer = 0;
	std::vector<GasHandle> m_gases;

	OptixModule m_raygen_module          = 0;
	OptixModule m_geometry_sphere_module = 0;
	OptixModule m_geometry_nvdb_module   = 0;
	OptixModule m_shading_module         = 0;

	OptixProgramGroup       m_raygen_prog_group                          = 0;
	OptixProgramGroup       m_miss_prog_group[RAY_TYPE_COUNT]            = {0, 0};
	OptixProgramGroup       m_hitgroup_mesh_prog_group[RAY_TYPE_COUNT]   = {0, 0};
	OptixProgramGroup       m_hitgroup_sphere_prog_group[RAY_TYPE_COUNT] = {0, 0};
	OptixProgramGroup       m_hitgroup_nvdb_prog_group[RAY_TYPE_COUNT]   = {0, 0};
	OptixPipeline           m_pipeline                                   = 0;
	OptixShaderBindingTable m_sbt                                        = {};

	OptixModuleCompileOptions   m_module_compile_options   = {};
	OptixPipelineCompileOptions m_pipeline_compile_options = {};
	OptixPipelineLinkOptions    m_pipeline_link_options    = {};

	const int maxTraceDepth = 2;

	// -- Denoiser
	bool                initializedDenoiser   = false;
	OptixDenoiser       m_denoiser            = nullptr;
	CUstream            m_denoiserStream      = 0;
	OptixDenoiserParams m_denoiserParams      = {};
	uint32_t            m_denoiserWidth       = 0;
	uint32_t            m_denoiserHeight      = 0;
	CUdeviceptr         m_denoiserIntensity   = 0;
	CUdeviceptr         m_denoiserScratch     = 0;
	uint32_t            m_denoiserScratchSize = 0;
	CUdeviceptr         m_denoiserState       = 0;
	uint32_t            m_denoiserStateSize   = 0;

	OptixImage2D m_denoiserInputs[3] = {};
	OptixImage2D m_denoiserOutput;

	// -- Interface
	void create();
	/**
	 * Call this before rendering. This will update the optix state and make it ready for rendering
	 * @return True if successful without errors and rendering can proceed
	 */
	bool preRender(nori::Scene &scene, bool usePreview);

	/**
	 * Initialize the denoiser with the current image dimensions and IO buffer device pointers
	 */
	void preRenderDenoiser(const uint32_t imageWidth, const uint32_t imageHeight,
	                       const float4 *d_composite, const float4 *d_albedo, const float4 *d_normal,
	                       float4 *d_denoised);
	/**
	 * Renders one subframe. Assumes that preRender has succeeded
	 * @param outputBuffer mapped device pointer, use CUDAOutputBuffer::map() on the opengl thread
	 */
	void renderSubframe(const uint32_t currentSample,
	                    float4 *outputBuffer, float4 *outputBufferAlbedo, float4 *outputBufferNormal);
	/**
	 * Denoises the device buffers, which should already be allocated
	 */
	void denoise();

	void clear();
	void clearPipeline();
	~OptixState();

private:
	void createContext();
	void createCompileOptions();
	void buildGases(const std::vector<nori::Shape *> &shapes);
	void buildIas();
	void createPtxModules(bool specialize = true);
	void createPipeline();
	void createRaygenProgram(std::vector<OptixProgramGroup> &program_groups);
	void createHitMeshProgram(std::vector<OptixProgramGroup> &program_groups);
	void createHitSphereProgram(std::vector<OptixProgramGroup> &program_groups);
	void createHitVolumeProgram(std::vector<OptixProgramGroup> &program_groups);
	void createMissProgram(std::vector<OptixProgramGroup> &program_groups);
	void updateSbt(const std::vector<nori::Shape *> &shapes);

	void deleteDenoiser();
};

#endif