//
// Created by roger on 06/12/2020.
//

/**
 * This file contains all acceleration structure (as) parts for OptixState
 * It creates and updates the GASes and IAS
 */

#include <nori/optix/OptixState.h>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <nvrtc.h>
#include <cuda_runtime_api.h>

#include "sutil/Exception.h"

#include <nori/shape.h>
#include <nori/mesh.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>
#include <algorithm>

struct GasBuildInfo
{
	OptixBuildInput       buildInput;
	OptixAccelBufferSizes gasBufferSizes;
	GasHandle             *gasHandle;

	bool operator<(GasBuildInfo &other) const
	{
		return gasBufferSizes.outputSizeInBytes > other.gasBufferSizes.outputSizeInBytes;
	}
};

/**
 * Creates a GAS for each nori::Shape.
 * Sets: m_gases
 */
void OptixState::buildGases(const std::vector<nori::Shape *> &shapes)
{
	const auto t0 = std::chrono::high_resolution_clock::now();
	if (initializedState)
	{
		for (auto &gas : m_gases)
			CUDA_CHECK(cudaFree(reinterpret_cast<void *>(gas.d_buffer)));
		m_gases.clear();
	}
	m_gases.resize(shapes.size());

	OptixAccelBuildOptions accelBuildOptions = {};
	accelBuildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	accelBuildOptions.buildFlags =
			OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;

	// -- Create a GASInfo per mesh and accumulate buffer sizes
	OptixAccelBufferSizes totalBufferSizes = {};
	memset(&totalBufferSizes, 0, sizeof(OptixAccelBufferSizes));

	// Accumulate build inputs from nori shapes and calculate the totalBufferSizes
	std::vector<GasBuildInfo> gasInfos{};
	gasInfos.reserve(shapes.size());

	// Stores the pointers of all 'new' allocated flag arrays
	std::vector<uint32_t*> buildInputFlags{shapes.size(), nullptr};

	for (int i = 0; i < shapes.size(); ++i)
	{
		// Get shape specific build input
		OptixBuildInput buildInput = shapes[i]->getOptixBuildInput(buildInputFlags[i]);

		OptixAccelBufferSizes gasBufferSizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context,
		                                         &accelBuildOptions,
		                                         &buildInput, 1,
		                                         &gasBufferSizes));

		totalBufferSizes.tempSizeInBytes += gasBufferSizes.tempSizeInBytes;
		totalBufferSizes.outputSizeInBytes += gasBufferSizes.outputSizeInBytes;
		totalBufferSizes.tempUpdateSizeInBytes += gasBufferSizes.tempUpdateSizeInBytes;

		GasBuildInfo gasBuildInfo = {buildInput, gasBufferSizes, &m_gases[i]};
		gasInfos.emplace_back(gasBuildInfo);
	}

	// -- Build all GasInfos and compact
	// sort by gasInfos outputBufferSize for efficiency
	std::sort(gasInfos.begin(), gasInfos.end());

	// Buffer are re-used
	size_t      tempBufferSize       = 0;
	CUdeviceptr d_tempBuffer         = 0;
	size_t      tempOutputBufferSize = 0;
	CUdeviceptr d_tempOutputBuffer   = 0;

	// Get compacted size property when building for the first time
	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&emitProperty.result), sizeof(size_t)));

	// Iterate over meshes from largest to smallest
	for (size_t i = 0; i < gasInfos.size(); ++i)
	{
		GasHandle &gasHandle = *gasInfos[i].gasHandle;

		// Re-alloc if temp (output) buffer is much smaller or larger than the current
		if (gasInfos[i].gasBufferSizes.tempSizeInBytes * 1.25f < tempBufferSize ||
		    gasInfos[i].gasBufferSizes.tempSizeInBytes > tempBufferSize)
		{
			CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));
			tempBufferSize = gasInfos[i].gasBufferSizes.tempSizeInBytes;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tempBuffer), tempBufferSize));
		}

		if (gasInfos[i].gasBufferSizes.outputSizeInBytes * 1.25f < tempOutputBufferSize
		    || gasInfos[i].gasBufferSizes.outputSizeInBytes > tempOutputBufferSize)
		{
			CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_tempOutputBuffer)));
			tempOutputBufferSize = gasInfos[i].gasBufferSizes.outputSizeInBytes;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tempOutputBuffer), tempOutputBufferSize));
		}

		// Initial accel build
		OPTIX_CHECK(optixAccelBuild(m_context, nullptr,
		                            &accelBuildOptions,
		                            &gasInfos[i].buildInput, 1,
		                            d_tempBuffer, tempBufferSize,
		                            d_tempOutputBuffer, tempOutputBufferSize,
		                            &gasHandle.handle,
		                            &emitProperty, 1));

		// Retrieve compacted size from device
		size_t compactedSize;
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(&compactedSize), reinterpret_cast<void *>(emitProperty.result),
		                      sizeof(size_t), cudaMemcpyDeviceToHost));

		// Check if compaction is worthwhile
		if (compactedSize >= gasInfos[i].gasBufferSizes.outputSizeInBytes)
		{
			tempOutputBufferSize = 0;
			gasInfos[i].gasHandle->d_buffer = d_tempOutputBuffer;
			continue;
		}

		// Allocate the compacted buffer and build
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gasHandle.d_buffer), compactedSize));

		OPTIX_CHECK(optixAccelCompact(m_context, nullptr,
		                              gasHandle.handle,
		                              gasHandle.d_buffer, compactedSize,
		                              &gasHandle.handle
		));
	}

	// Be sure to delete the 'new' allocated arrays
	for(uint32_t* flagArray : buildInputFlags)
		delete flagArray;

	const auto                    t1       = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = t1 - t0;
	std::cout << "Optix: Built " << gasInfos.size() << " GASes in: " << duration.count() << "ms\n";
}

/**
 * Re/builds the IAS using the already created GASes.
 * Rebuild will delete the ias and create a new one.
 * requires: m_gases
 * sets: m_d_ias_output_buffer, m_ias_handle
 */
void OptixState::buildIas()
{
	const auto t0 = std::chrono::high_resolution_clock::now();
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_ias_output_buffer)));
	m_d_ias_output_buffer = 0;

	const uint32_t             numInstances = static_cast<uint32_t>(m_gases.size());
	std::vector<OptixInstance> optixInstances(numInstances);

	const float identityTransform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
	uint32_t    sbtOffset             = 0;

	for (uint32_t i = 0; i < numInstances; ++i)
	{
		auto gasHandle      = m_gases[i];
		auto &optixInstance = optixInstances[i];
		memset(&optixInstance, 0, sizeof(OptixInstance));

		optixInstance.flags             = OPTIX_INSTANCE_FLAG_NONE;
		optixInstance.instanceId        = i;
		optixInstance.sbtOffset         = sbtOffset;
		optixInstance.visibilityMask    = 1;
		optixInstance.traversableHandle = gasHandle.handle;
		memcpy(optixInstance.transform, identityTransform, sizeof(float) * 12);

		sbtOffset += RAY_TYPE_COUNT;
	}

	// Copy the instances to the device
	const size_t instancesSizeBytes = sizeof(OptixInstance) * numInstances;
	CUdeviceptr  d_tempInstances;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tempInstances), instancesSizeBytes));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_tempInstances), optixInstances.data(), instancesSizeBytes,
	                      cudaMemcpyHostToDevice));

	OptixBuildInput instanceBuildInput = {};
	instanceBuildInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instanceBuildInput.instanceArray.instances    = d_tempInstances;
	instanceBuildInput.instanceArray.numInstances = numInstances;

	OptixAccelBuildOptions accelBuildOptions = {};
	accelBuildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;
	accelBuildOptions.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

	// Compute buffer sizes required first, allocate them and then build the AS
	OptixAccelBufferSizes iasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context,
	                                         &accelBuildOptions,
	                                         &instanceBuildInput, 1,
	                                         &iasBufferSizes));

	CUdeviceptr d_tempBuffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_tempBuffer), iasBufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_ias_output_buffer)));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_ias_output_buffer), iasBufferSizes.outputSizeInBytes));

	OPTIX_CHECK(optixAccelBuild(m_context, nullptr, // device context + cuda stream
	                            &accelBuildOptions,
	                            &instanceBuildInput, 1,
	                            d_tempBuffer, iasBufferSizes.tempSizeInBytes,
	                            m_d_ias_output_buffer, iasBufferSizes.outputSizeInBytes,
	                            &m_ias_handle,
	                            nullptr, 0 // no emitted properties
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_tempInstances)));

	const auto                    t1       = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = t1 - t0;
	std::cout << "Built IAS in: " << duration.count() << "s\n";
}
