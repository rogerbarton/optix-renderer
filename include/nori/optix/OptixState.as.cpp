//
// Created by roger on 06/12/2020.
//

/**
 * This file contains all acceleration structure (as) parts for OptixState
 * It creates and updates the GASes and IAS
 */

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
 */
void OptixState::buildGases(std::vector<nori::Shape *> &shapes)
{
	const auto t0 = std::chrono::high_resolution_clock::now();

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
	for (int i = 0; i < shapes.size(); ++i)
	{
		// Get shape specific build input
		OptixBuildInput buildInput = shapes[i]->getOptixBuildInput();

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

	const auto                    t1       = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = t1 - t0;
	std::cout << "Optix: Built " << gasInfos.size() << " GASes in: " << duration.count() << "ms\n";
}

/**
 * GasInfo for a triangle mesh
 */
OptixBuildInput nori::Mesh::getOptixBuildInput() const
{
	// Copy mesh data to device
	// TODO: delete this, store this in mesh?
	CUdeviceptr d_V;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &d_V ), sizeof(OptixAabb)));
	CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void *>( d_V ),
			m_V.data(), m_V.size() * sizeof(float),
			cudaMemcpyHostToDevice
	));

	CUdeviceptr d_F;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &d_F ), sizeof(OptixAabb)));
	CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void *>( d_F ),
			m_F.data(), m_F.size() * sizeof(float),
			cudaMemcpyHostToDevice
	));

	OptixBuildInput buildInputs = {};

	buildInputs.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	const uint32_t flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
	buildInputs.triangleArray.flags         = flags;
	buildInputs.triangleArray.numSbtRecords = 1;

	buildInputs.triangleArray.vertexBuffers       = &d_V;
	buildInputs.triangleArray.numVertices         = m_V.cols();
	buildInputs.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
	buildInputs.triangleArray.vertexStrideInBytes = m_V.stride() ? m_V.stride() : sizeof(float3);

	buildInputs.triangleArray.indexBuffer        = d_F;
	buildInputs.triangleArray.numIndexTriplets   = m_F.cols();
	buildInputs.triangleArray.indexFormat        = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	buildInputs.triangleArray.indexStrideInBytes = m_F.stride() ? m_F.stride() : sizeof(uint32_t) * 3;

	return buildInputs;
}

/**
 * Default is to use bbox with custom intersection
 */
OptixBuildInput nori::Shape::getOptixBuildInput() const
{
	// AABB build input
	OptixAabb aabb = {m_bbox.min.x(), m_bbox.min.y(), m_bbox.min.z(),
	                  m_bbox.max.x(), m_bbox.max.y(), m_bbox.max.z()};

	// TODO: delete this
	CUdeviceptr d_aabb_buffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>( &d_aabb_buffer ), sizeof(OptixAabb)));
	CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void *>( d_aabb_buffer ),
			&aabb,
			sizeof(OptixAabb),
			cudaMemcpyHostToDevice
	));

	uint32_t aabb_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

	OptixBuildInput buildInput = {};
	buildInput.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	buildInput.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
	buildInput.customPrimitiveArray.numPrimitives = 1;
	buildInput.customPrimitiveArray.flags         = aabb_input_flags;
	buildInput.customPrimitiveArray.numSbtRecords = 1;

	return buildInput;
}

void OptixState::buildIas()
{

}
