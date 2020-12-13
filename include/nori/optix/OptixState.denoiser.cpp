//
// Created by roger on 13/12/2020.
//

#include "OptixState.h"

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

/**
 * Based on OptiX samples
 */

void OptixState::preRenderDenoiser(const uint32_t imageWidth, const uint32_t imageHeight, const float4 *d_composite,
                                   const float4 *d_albedo, const float4 *d_normal,
                                   float4 *d_denoised)
{
	/**
	 * There are three stages:
	 * 1. One time creation of the denoiser
	 * 2. Setup of the denoiser with the current image dimensions
	 * 3. Binding of the IO feature buffers
	 * 4. Executing the denoiser -> denoise()
	 */

	// Create denoiser
	if (!initializedDenoiser)
	{
		initializedDenoiser = true;
		CUDA_CHECK(cudaStreamCreate(&m_denoiserStream));

		OptixDenoiserOptions options = {};
		options.inputKind = OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
		OPTIX_CHECK(optixDenoiserCreate(m_context, &options, &m_denoiser));
		OPTIX_CHECK(optixDenoiserSetModel(
				m_denoiser,
				OPTIX_DENOISER_MODEL_KIND_HDR,
				nullptr, // data
				0        // size
		));
	}

	// Allocate device memory for denoiser and setup
	if (imageWidth != m_denoiserWidth || imageHeight != m_denoiserHeight)
	{
		// delete old state
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_denoiserIntensity)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_denoiserScratch)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_denoiserState)));

		m_denoiserWidth  = imageWidth;
		m_denoiserHeight = imageHeight;

		OptixDenoiserSizes denoiserSizes;
		OPTIX_CHECK(optixDenoiserComputeMemoryResources(
				m_denoiser,
				m_denoiserWidth,
				m_denoiserHeight,
				&denoiserSizes
		));

		// NOTE: if using tiled denoising, we would set scratch-size to
		//       denoiser_sizes.withOverlapScratchSizeInBytes
		m_denoiserScratchSize = static_cast<uint32_t>( denoiserSizes.withoutOverlapScratchSizeInBytes );

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_denoiserIntensity), sizeof(float)));

		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_denoiserScratch), m_denoiserScratchSize));

		m_denoiserStateSize = static_cast<uint32_t>(denoiserSizes.stateSizeInBytes);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_denoiserState), m_denoiserStateSize));

		// -- Setup denoiser
		OPTIX_CHECK(optixDenoiserSetup(
				m_denoiser,
				m_denoiserStream,  // CUDA stream
				m_denoiserWidth,
				m_denoiserHeight,
				m_denoiserState,
				m_denoiserStateSize,
				m_denoiserScratch,
				m_denoiserScratchSize
		));

		m_denoiserParams.denoiseAlpha = 0;
		m_denoiserParams.hdrIntensity = m_denoiserIntensity;
		m_denoiserParams.blendFactor  = 0.0f;
	}

	// -- Bind IO image buffers
	{
		m_denoiserInputs[0].data               = (CUdeviceptr) d_composite;
		m_denoiserInputs[0].width              = m_denoiserWidth;
		m_denoiserInputs[0].height             = m_denoiserHeight;
		m_denoiserInputs[0].rowStrideInBytes   = m_denoiserWidth * sizeof(float4);
		m_denoiserInputs[0].pixelStrideInBytes = sizeof(float4);
		m_denoiserInputs[0].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

		m_denoiserInputs[1].data               = (CUdeviceptr) d_albedo;
		m_denoiserInputs[1].width              = m_denoiserWidth;
		m_denoiserInputs[1].height             = m_denoiserHeight;
		m_denoiserInputs[1].rowStrideInBytes   = m_denoiserWidth * sizeof(float4);
		m_denoiserInputs[1].pixelStrideInBytes = sizeof(float4);
		m_denoiserInputs[1].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

		m_denoiserInputs[2].data               = (CUdeviceptr) d_normal;
		m_denoiserInputs[2].width              = m_denoiserWidth;
		m_denoiserInputs[2].height             = m_denoiserHeight;
		m_denoiserInputs[2].rowStrideInBytes   = m_denoiserWidth * sizeof(float4);
		m_denoiserInputs[2].pixelStrideInBytes = sizeof(float4);
		m_denoiserInputs[2].format             = OPTIX_PIXEL_FORMAT_FLOAT4;

		m_denoiserOutput.data               = (CUdeviceptr) d_denoised;
		m_denoiserOutput.width              = m_denoiserWidth;
		m_denoiserOutput.height             = m_denoiserHeight;
		m_denoiserOutput.rowStrideInBytes   = m_denoiserWidth * sizeof(float4);
		m_denoiserOutput.pixelStrideInBytes = sizeof(float4);
		m_denoiserOutput.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
	}
}

void OptixState::denoise()
{

	OPTIX_CHECK(optixDenoiserComputeIntensity(
			m_denoiser,
			m_denoiserStream, // CUDA stream
			m_denoiserInputs,
			m_denoiserIntensity,
			m_denoiserScratch,
			m_denoiserScratchSize
	));

	OPTIX_CHECK(optixDenoiserInvoke(
			m_denoiser,
			m_denoiserStream, // CUDA stream
			&m_denoiserParams,
			m_denoiserState,
			m_denoiserStateSize,
			m_denoiserInputs,
			3, // num input channels
			0, // input offset X
			0, // input offset y
			&m_denoiserOutput,
			m_denoiserScratch,
			m_denoiserScratchSize
	));

	CUDA_CHECK(cudaStreamSynchronize(m_denoiserStream)); // Sync for now
	CUDA_SYNC_CHECK();
}

void OptixState::deleteDenoiser()
{

	OPTIX_CHECK(optixDenoiserDestroy(m_denoiser));

	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_denoiserIntensity)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_denoiserScratch)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_denoiserState)));

	cudaStreamDestroy(m_denoiserStream);
}
