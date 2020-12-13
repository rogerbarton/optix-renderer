/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob, Romain Prévost

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/render.h>
#include <nori/parser.h>
#include <nori/scene.h>
#include <nori/camera.h>
#include <nori/block.h>
#include <nori/timer.h>
#include <nori/bitmap.h>
#include <nori/sampler.h>
#include <nori/integrator.h>

#include <nori/ImguiHelpers.h>
#include <nori/rfilter.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_scheduler_init.h>
#include <filesystem/resolver.h>
#include <fstream>


#ifdef NORI_USE_OPTIX
#include <nori/optix/OptixState.h>
#endif

NORI_NAMESPACE_BEGIN

static void renderBlock(const Scene *const scene, Integrator *const integrator, Sampler *const sampler,
						ImageBlock &block, ImageBlock &blockAlbedo, ImageBlock &blockNormals);

RenderThread::RenderThread() : m_startTime(clock_t::now()), m_endTime(m_startTime), sceneFilename(""),
                               m_block{ImageBlock(Vector2i(720, 720))},
                               m_blockAlbedo{ImageBlock(Vector2i(720, 720))},
                               m_blockNormal{ImageBlock(Vector2i(720, 720))}
{
}


void RenderThread::initBlocks()
{
#ifdef NORI_USE_OPTIX
	m_optixDisplayBlock       = new CUDAOutputBuffer <float4>{CUDAOutputBufferType::GL_INTEROP, 1, 1};
	m_optixDisplayBlockAlbedo = new CUDAOutputBuffer <float4>{CUDAOutputBufferType::GL_INTEROP, 1, 1};
	m_optixDisplayBlockNormal = new CUDAOutputBuffer <float4>{CUDAOutputBufferType::GL_INTEROP, 1, 1};
	m_optixDisplayBlockDenoised = new CUDAOutputBuffer <float4>{CUDAOutputBufferType::GL_INTEROP, 1, 1};
#endif
	m_block.setConstant(Color4f(0.6f, 0.6f, 0.6f, 1.00f));
}

RenderThread::~RenderThread()
{
	stopRendering();
	delete m_guiScene;
	delete m_renderScene;
#ifdef NORI_USE_OPTIX
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlock)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlockAlbedo)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlockNormal)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlockDenoised)));
	delete m_optixDisplayBlock;
	delete m_optixDisplayBlockAlbedo;
	delete m_optixDisplayBlockNormal;
	delete m_optixDisplayBlockDenoised;
#endif
}


void RenderThread::PreGlDestroy()
{
#ifdef NORI_USE_OPTIX
	m_optixDisplayBlock->deletePBO();
	m_optixDisplayBlockAlbedo->deletePBO();
	m_optixDisplayBlockNormal->deletePBO();
	m_optixDisplayBlockDenoised->deletePBO();
#endif
}

bool RenderThread::isBusy()
{
	if (m_renderStatus == ERenderStatus::Done)
	{
		m_renderThread.join();
		m_renderStatus = ERenderStatus::Idle;
	}
	return m_renderStatus != ERenderStatus::Idle;
}

void RenderThread::stopRendering()
{
	if (isBusy())
	{
		cout << "Requesting interruption of the current rendering" << endl;
		m_renderStatus = ERenderStatus::Interrupt;
		m_renderThread.join();
		m_renderStatus = ERenderStatus::Idle;
		cout << "Rendering successfully aborted" << endl;
	}
}

void RenderThread::loadScene(const std::string &filename)
{
	// Trigger interrupt, but don't wait
	if (isBusy())
		m_renderStatus = ERenderStatus::Interrupt;

	filesystem::path path(filename);

	/* Add the parent directory of the scene file to the
		   file resolver. That way, the XML file can reference
		   resources (OBJ files, textures) using relative paths */
	getFileResolver()->prepend(path.parent_path());

	NoriObject *root = loadFromXML(filename);

	if (root->getClassType() != NoriObject::EScene)
	{
		delete root;
		return;
	}

	// Wait for current render to finish before continuing
	if (isBusy())
	{
		m_renderThread.join();
		m_renderStatus = ERenderStatus::Idle;
	}

	// -- Accept new scene and initialize it
	sceneFilename = filename;

	// Delete old scene if exists
	if (m_guiScene)
	{
		delete m_guiScene;
		m_guiScene = nullptr;
		delete m_renderScene;
		m_renderScene = nullptr;
	}

	// Create gui/properties scene first, then deep copy to render scene
	m_guiScene = static_cast<Scene *>(root);
	m_renderScene = static_cast<Scene *>(m_guiScene->cloneAndInit());

	// Determine the filename of the output bitmap
	outputName = sceneFilename;
	size_t lastdot = outputName.find_last_of(".");
	if (lastdot != std::string::npos)
		outputName.erase(lastdot, std::string::npos);

	outputNameVariance = outputName + "_variance.exr";
	outputName += ".exr";

	m_renderScene->update(m_guiScene);

	// Reset the render layer if it is not supported, assuming preview integrator supports all layers
	if ((m_visibleRenderLayer & m_renderScene->getIntegrator(false)->getSupportedLayers()) == 0)
		m_visibleRenderLayer = ERenderLayer::Composite;

	startRenderThread();
}

void RenderThread::restartRender()
{
	m_guiSceneTouched = false;
	if (!m_guiScene)
		return;

	stopRendering();

	m_renderScene->update(m_guiScene);

	startRenderThread();
}

void RenderThread::startRenderThread()
{
	// Resize block on main thread
	const Vector2i outputSize = m_renderScene->getCamera()->getOutputSize();
#ifdef NORI_USE_OPTIX
	m_optixDisplayBlock->lock();
	if (m_d_optixDisplayBlock)
		m_optixDisplayBlock->unmap();
	m_optixDisplayBlock->resize(outputSize.x(), outputSize.y());
	m_d_optixDisplayBlock = m_optixDisplayBlock->map();
	m_optixDisplayBlock->unlock();

	m_optixDisplayBlockAlbedo->lock();
	if (m_d_optixDisplayBlockAlbedo)
		m_optixDisplayBlockAlbedo->unmap();
	m_optixDisplayBlockAlbedo->resize(outputSize.x(), outputSize.y());
	m_d_optixDisplayBlockAlbedo = m_optixDisplayBlockAlbedo->map();
	m_optixDisplayBlockAlbedo->unlock();

	m_optixDisplayBlockNormal->lock();
	if (m_d_optixDisplayBlockNormal)
		m_optixDisplayBlockNormal->unmap();
	m_optixDisplayBlockNormal->resize(outputSize.x(), outputSize.y());
	m_d_optixDisplayBlockNormal = m_optixDisplayBlockNormal->map();
	m_optixDisplayBlockNormal->unlock();

	m_optixDisplayBlockDenoised->lock();
	if (m_d_optixDisplayBlockDenoised)
		m_optixDisplayBlockDenoised->unmap();
	m_optixDisplayBlockDenoised->resize(outputSize.x(), outputSize.y());
	m_d_optixDisplayBlockDenoised = m_optixDisplayBlockDenoised->map();
	m_optixDisplayBlockDenoised->unlock();
#endif

	// Start the actual thread
	m_renderStatus = ERenderStatus::Busy;
	m_renderThread = std::thread([this] { renderThreadMain(); });
}

void RenderThread::renderThreadMain()
{
	m_startTime = clock_t::now();
	m_endTime = time_point_t::min();
	cout << "Rendering .. " << std::flush;

	/* Allocate memory for the entire output image and clear it */
	const Camera * const camera = m_renderScene->getCamera();
	const Vector2i outputSize = camera->getOutputSize();
	m_block.lock();
	m_block.init(outputSize, camera->getReconstructionFilter());
	m_block.clear();
	m_block.unlock();
	m_blockAlbedo.lock();
	m_blockAlbedo.init(outputSize, camera->getReconstructionFilter());
	m_blockAlbedo.clear();
	m_blockAlbedo.unlock();
	m_blockNormal.lock();
	m_blockNormal.init(outputSize, camera->getReconstructionFilter());
	m_blockNormal.clear();
	m_blockNormal.unlock();

	m_currentCpuSample = 0;
	m_currentOptixSample = 0;
#ifdef NORI_USE_OPTIX
	if (m_renderDevice != EDeviceMode::Cpu)
	{
		tbb::task_scheduler_init init(tbb::task_scheduler_init::default_num_threads() - 2);
		m_optixThread = std::thread([this] { renderThreadOptix(); });
	}
#endif

	/* Create a block generator (i.e. a work scheduler) */
	const int blockSize = m_renderScene->getSampler()->isAdaptive() ? NORI_BLOCK_SIZE_ADAPTIVE : NORI_BLOCK_SIZE;
	BlockGenerator blockGenerator(outputSize, blockSize);

	Timer timer;

	Integrator *const integrator = m_renderScene->getIntegrator(m_previewMode);
	integrator->preprocess(m_renderScene);

	const uint32_t numSamples = m_previewMode ? 1 : m_renderScene->getSampler()->getSampleCount();
	const auto numBlocks = blockGenerator.getBlockCount();

	tbb::concurrent_vector<std::unique_ptr<Sampler>> samplers;
	samplers.resize(numBlocks);

	std::cout << std::endl;
	for (; m_currentCpuSample + m_currentOptixSample < numSamples; ++m_currentCpuSample)
	{
		const uint32_t currentSample = m_currentCpuSample;
		m_progress = currentSample / float(numSamples);
		if (m_renderStatus == ERenderStatus::Interrupt)
			break;

		tbb::blocked_range<int> range(0, numBlocks);

		m_renderScene->getSampler()->setSampleRound(currentSample);

		auto map = [&](const tbb::blocked_range<int> &range) {
			// Allocate memory for a small image block to be rendered by the current thread
			ImageBlock block(Vector2i(blockSize), camera->getReconstructionFilter());
			ImageBlock blockAlbedo(Vector2i(blockSize), camera->getReconstructionFilter());
			ImageBlock blockNormals(Vector2i(blockSize), camera->getReconstructionFilter());

			for (int i = range.begin(); i < range.end(); ++i)
			{
				// i = index of current rendering block
				if (m_renderStatus == ERenderStatus::Interrupt)
					break;

				// Request an image block from the block generator
				blockGenerator.next(block);
				blockAlbedo.setOffset(block.getOffset());
				blockAlbedo.setSize(block.getSize());
				blockAlbedo.setBlockId(block.getBlockId());
				blockNormals.setOffset(block.getOffset());
				blockNormals.setSize(block.getSize());
				blockNormals.setBlockId(block.getBlockId());

				// Get block id to continue using the same sampler
				auto blockId = block.getBlockId();
				if (currentSample == 0)
				{ // Initialize the sampler for the first sample
					std::unique_ptr<Sampler> sampler(m_renderScene->getSampler()->clone());
					sampler->prepare(block);
					samplers.at(blockId) = std::move(sampler);
				}

				// this gets executed if uniform or adaptive and we need to render
				if (samplers.at(blockId)->isAdaptive())
				{
					// while we have to recompute this block, we do it...
					int count = 0;
					samplers.at(blockId)->setSampleRound(count); // needed for computeVariance
					while (samplers.at(blockId)->computeVariance(block))
					{
						renderBlock(m_renderScene, integrator, samplers.at(blockId).get(), block, blockAlbedo, blockNormals);
						m_block.put(block); // save to master block / live view of rendering
						m_blockAlbedo.put(blockAlbedo);
						m_blockNormal.put(blockNormals);
						samplers.at(blockId)->setSampleRound(count++);
					}
				}
				else
				{
					// for uniform sampling, simply call render on the block
					renderBlock(m_renderScene, integrator, samplers.at(blockId).get(), block, blockAlbedo, blockNormals);
					m_block.put(block); // save to master block
					m_blockAlbedo.put(blockAlbedo);
					m_blockNormal.put(blockNormals);
				}
			}
		};
		tbb::parallel_for(range, map);

		if (m_renderStatus == ERenderStatus::Interrupt)
			break;

		// do these in serial, not parallel (potential race condition)
		for (int i = 0; i < samplers.size(); i++)
		{
			m_renderScene->getSampler()->addToTotalSamples(samplers.at(i)->getTotalSamples());
			samplers.at(i)->setTotalSamples(0);
		}

		blockGenerator.reset();
	}

#ifdef NORI_USE_OPTIX
	if(m_optixThread.joinable())
		m_optixThread.join();
#endif

	if (m_renderScene->getDenoiser())
		m_renderScene->getDenoiser()->denoise(&m_block);

	m_endTime = clock_t::now();
	cout << "done. (took " << timer.elapsedString() << ")" << endl;
	cout << "Total Samples Placed: " << m_renderScene->getSampler()->getTotalSamples() << std::endl;
	if (m_previewMode || m_renderStatus == ERenderStatus::Interrupt)
	{
		// stop the rendering here, don't save
		m_renderStatus = ERenderStatus::Done;
		return;
	}

	/* Now turn the rendered image block into
		   a properly normalized bitmap */
	m_block.lock();
	std::unique_ptr<Bitmap> bitmap(m_block.toBitmap());
	m_block.unlock();

	/* Save using the OpenEXR format */
	bitmap->save(outputName);

	// write variance to disk
	// for now, disable variance writer
	if (m_renderScene->getSampler()->isAdaptive())
	{
		BlockGenerator       blockGen(outputSize, blockSize);
		ReconstructionFilter *rf = static_cast<ReconstructionFilter *>(NoriObjectFactory::createInstance("box"));
		ImageBlock           currVarBlock(Vector2i(blockSize), rf);

		ImageBlock fullVarianceMatrix(outputSize, rf);
		fullVarianceMatrix.clear();
		const int blocks = blockGen.getBlockCount();

		for (int i = 0; i < blocks; i++)
		{
			blockGen.next(currVarBlock);
			int id = currVarBlock.getBlockId();
			currVarBlock.clear();
			samplers.at(id)->writeVarianceMatrix(currVarBlock);

			fullVarianceMatrix.put(currVarBlock);
		}

		fullVarianceMatrix.toBitmap()->save(outputNameVariance);

		delete rf;
	}
	std::cout << "Mean variance of m_block: " << computeVarianceFromImage(m_block).mean() << std::endl;

	m_renderStatus = ERenderStatus::Done;
}

static void renderBlock(const Scene *const scene, Integrator *const integrator, Sampler *const sampler,
                        ImageBlock &block, ImageBlock &blockAlbedo, ImageBlock &blockNormal)
{
	const Camera *camera = scene->getCamera();

	Point2i offset = block.getOffset();

	/* For each pixel and pixel sample sample */
	std::vector<std::pair<int, int>> sampleIndices = sampler->getSampleIndices(block);

	/* Clear the block contents */
	block.clear();
	blockAlbedo.clear();
	blockNormal.clear();

	for (int i = 0; i < sampleIndices.size(); i++)
	{
		std::pair<int, int> pair = sampleIndices[i];
		int x = pair.first;
		int y = pair.second;
		Point2f pixelSample = Point2f((float)(x + offset.x()), (float)(y + offset.y())) +
							  sampler->next2D();
		Point2f apertureSample = sampler->next2D();

		/* Sample a ray from the camera */
		Ray3f ray;
		Color3f value = camera->sampleRay(ray, pixelSample, apertureSample);

		/* Compute the incident radiance */
		Color3f albedo = 0.f;
		Color3f normal = 0.f;
		value *= integrator->Li(scene, sampler, ray, albedo, normal);

		/* Store in the image block */
		block.put(pixelSample, value);
		blockAlbedo.put(pixelSample, albedo);
		blockNormal.put(pixelSample, normal);
	}
}

#ifdef NORI_USE_OPTIX
void RenderThread::renderThreadOptix()
{
	try
	{
		// Resize blocks
		const Vector2i blockSize      = m_renderScene->getCamera()->getOutputSize();
		const size_t   blockSizeBytes = blockSize.x() * blockSize.y() * sizeof(float4);;
		if (m_optixBlockSizeBytes != blockSizeBytes)
		{
			m_optixBlockSizeBytes = blockSizeBytes;
			CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlock)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlockAlbedo)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlockNormal)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_optixRenderBlockDenoised)));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_optixRenderBlock), m_optixBlockSizeBytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_optixRenderBlockAlbedo), m_optixBlockSizeBytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_optixRenderBlockNormal), m_optixBlockSizeBytes));
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_optixRenderBlockDenoised), m_optixBlockSizeBytes));
			CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(m_d_optixRenderBlock), 0, m_optixBlockSizeBytes));
			CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(m_d_optixRenderBlockAlbedo), 0, m_optixBlockSizeBytes));
			CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(m_d_optixRenderBlockNormal), 0, m_optixBlockSizeBytes));
			CUDA_CHECK(cudaMemset(reinterpret_cast<void *>(m_d_optixRenderBlockDenoised), 0, m_optixBlockSizeBytes));
		}

		OptixState *const optixState = m_renderScene->getOptixState();

		if (!optixState->preRender(*m_renderScene, m_previewMode))
			return;

		optixState->preRenderDenoiser(blockSize.x(), blockSize.y(),
		                              m_d_optixRenderBlock, m_d_optixRenderBlockAlbedo, m_d_optixRenderBlockNormal,
		                              m_d_optixRenderBlockDenoised);

		// Render
		const uint32_t numSamples       = m_renderScene->getSampler()->getSampleCount();
		const uint32_t samplesPerLaunch = m_renderScene->m_optixRenderer->m_samplesPerLaunch;
		const uint32_t denoiseRate      = m_renderScene->m_optixRenderer->m_denoiseRate;

		for (; m_currentOptixSample + m_currentCpuSample < numSamples; m_currentOptixSample += samplesPerLaunch)
		{
			if (m_renderStatus == ERenderStatus::Interrupt)
				break;

			optixState->renderSubframe(m_currentOptixSample,
			                           m_d_optixRenderBlock, m_d_optixRenderBlockAlbedo, m_d_optixRenderBlockNormal);

			const bool denoise = denoiseRate != 0 && m_currentOptixSample > 0 && m_currentOptixSample % denoiseRate == 0;
			if (denoise)
				optixState->denoise();

			// Copy output to display buffers if display buffer is available/mapped
			if (m_d_optixDisplayBlock)
			{
				m_optixDisplayBlock->lock();
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_d_optixDisplayBlock),
				                      m_d_optixRenderBlock,
				                      m_optixBlockSizeBytes, cudaMemcpyDeviceToDevice));
				m_optixDisplayBlock->unlock();

				m_optixDisplayBlockAlbedo->lock();
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_d_optixDisplayBlockAlbedo),
				                      m_d_optixRenderBlockAlbedo,
				                      m_optixBlockSizeBytes, cudaMemcpyDeviceToDevice));
				m_optixDisplayBlockAlbedo->unlock();

				m_optixDisplayBlockNormal->lock();
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_d_optixDisplayBlockNormal),
				                      m_d_optixRenderBlockNormal,
				                      m_optixBlockSizeBytes, cudaMemcpyDeviceToDevice));
				m_optixDisplayBlockNormal->unlock();

				if (denoise) {
					m_optixDisplayBlockDenoised->lock();
					CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_d_optixDisplayBlockDenoised),
					                      m_d_optixRenderBlockDenoised,
					                      m_optixBlockSizeBytes, cudaMemcpyDeviceToDevice));
					m_optixDisplayBlockDenoised->unlock();

					m_optixDisplayBlockDenoisedTouched = true;
				}

				m_optixDisplayBlockTouched = true;
			}
		}

		// Denoise only at the end
		if (m_renderStatus != ERenderStatus::Interrupt && denoiseRate == 0)
		{
			optixState->denoise();

			// Update display buffer (same as above)
			m_optixDisplayBlockDenoised->lock();
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(m_d_optixDisplayBlockDenoised),
			                      m_d_optixRenderBlockDenoised,
			                      m_optixBlockSizeBytes, cudaMemcpyDeviceToDevice));
			m_optixDisplayBlockDenoised->unlock();

			m_optixDisplayBlockDenoisedTouched = true;
			m_optixDisplayBlockTouched = true;
		}
	}
	catch (std::exception& e)
	{
		std::cerr << "Optix Error: " << e.what() << std::endl;
		std::cerr << "-- Optix disabled.";
		// m_renderDevice  = EDeviceMode::Cpu;
		// m_displayDevice = EDeviceMode::Cpu;
	}
}

	void RenderThread::updateOptixDisplayBuffers()
	{
		if (!m_optixDisplayBlockTouched) return;
		m_optixDisplayBlockTouched = false;

		// Remap the display buffer so changes are applied to opengl
		m_optixDisplayBlock->lock();
		m_d_optixDisplayBlock = 0;
		m_optixDisplayBlock->unmap();
		m_d_optixDisplayBlock = m_optixDisplayBlock->map();
		m_optixDisplayBlock->unlock();

		m_optixDisplayBlockAlbedo->lock();
		m_d_optixDisplayBlockAlbedo = 0;
		m_optixDisplayBlockAlbedo->unmap();
		m_d_optixDisplayBlockAlbedo = m_optixDisplayBlockAlbedo->map();
		m_optixDisplayBlockAlbedo->unlock();

		m_optixDisplayBlockNormal->lock();
		m_d_optixDisplayBlockNormal = 0;
		m_optixDisplayBlockNormal->unmap();
		m_d_optixDisplayBlockNormal = m_optixDisplayBlockNormal->map();
		m_optixDisplayBlockNormal->unlock();

		if (m_optixDisplayBlockDenoisedTouched)
		{
			m_optixDisplayBlockDenoisedTouched = false;
			m_optixDisplayBlockDenoised->lock();
			m_d_optixDisplayBlockDenoised = 0;
			m_optixDisplayBlockDenoised->unmap();
			m_d_optixDisplayBlockDenoised = m_optixDisplayBlockDenoised->map();
			m_optixDisplayBlockDenoised->unlock();
		}
	}
#endif

#ifdef NORI_USE_IMGUI
void RenderThread::drawRenderGui()
{
	ImGui::Text("Render");
	ImGui::SameLine();
	ImGui::ProgressBar(getProgress());
	ImGui::Text("Render time: %s", (char*)getRenderTime().c_str());
	const uint32_t currentCpuSample = m_currentCpuSample;
	const uint32_t currentOptixSample = m_currentOptixSample;
	ImGui::Text("Samples Done: %i (cpu: %i, gpu: %i)", currentCpuSample + currentOptixSample, currentCpuSample, currentOptixSample);

	if (ImGui::Button("Stop Render"))
		stopRendering();

	// show restart button if m_scene is valid
	if (m_guiScene)
	{
		ImGui::SameLine();
		if (ImGui::Button("Restart Render"))
			restartRender();
	}

	ImGui::Checkbox("Auto-update", &m_autoUpdate);
	if (m_guiSceneTouched)
	{
		ImGui::SameLine();
		if (ImGui::Button("Apply Changes"))
			restartRender();
	}

	m_guiSceneTouched |= ImGui::Checkbox("Preview", &m_previewMode);
	m_guiSceneTouched |= ImGui::Combo("Device Mode", reinterpret_cast<int *>(&m_renderDevice), EDeviceMode::Strings, EDeviceMode::Size);

	ImGui::Separator();
	ImGui::Combo("Display Device", reinterpret_cast<int *>(&m_displayDevice), EDeviceMode::Strings, EDeviceMode::Size);
	ImGui::Combo("Visible Layer", reinterpret_cast<int *>(&m_visibleRenderLayer), ERenderLayer::Strings, ERenderLayer::Size);
}

void RenderThread::drawSceneGui()
{
	if (!m_guiScene)
	{
		ImGui::Text("No scene loaded...");
		return;
	}

	// Start columns
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
	ImGui::Columns(2);

	ImGui::AlignTextToFramePadding();
	ImGui::TreeNodeEx("fileName", ImGuiLeafNodeFlags, "Filename");
	ImGui::NextColumn();
	ImGui::SetNextItemWidth(-1);
	ImGui::Text(filesystem::path(sceneFilename).filename().c_str());
	ImGui::NextColumn();

	m_guiSceneTouched |= m_guiScene->getImGuiNodes();

	// end columns
	ImGui::Columns(1);
	ImGui::Separator();
	ImGui::PopStyleVar();

	if ((m_autoUpdate || m_previewMode) && m_guiSceneTouched)
		restartRender();
}
#endif

std::string RenderThread::getRenderTime() {
	if(m_endTime == m_startTime)
		return "-";
	else if (m_endTime > m_startTime)
		return timeString2(m_endTime - m_startTime);
	else
		return timeString2(clock_t::now() - m_startTime);
}

	ImageBlock &RenderThread::getBlock(ERenderLayer_t layer)
	{
		if(layer == ERenderLayer::Size)
			layer = m_visibleRenderLayer;

		if (layer == ERenderLayer::Albedo)
			return m_blockAlbedo;
		if (layer == ERenderLayer::Normal)
			return m_blockNormal;
		// if (layer == ERenderLayer::Denoised)
		// 	return m_blockDenoised;
		else
			return m_block;
	}

#ifdef NORI_USE_OPTIX
	CUDAOutputBuffer<float4> *RenderThread::getDisplayBlockGpu(ERenderLayer_t layer)
	{
		if(layer == ERenderLayer::Size)
			layer = m_visibleRenderLayer;

		if (layer == ERenderLayer::Albedo)
			return m_optixDisplayBlockAlbedo;
		if (layer == ERenderLayer::Normal)
			return m_optixDisplayBlockNormal;
		if (layer == ERenderLayer::Denoised)
			return m_optixDisplayBlockDenoised;
		else
			return m_optixDisplayBlock;
	}
#endif

void RenderThread::getDeviceSampleWeights(float &samplesCpu, float &samplesGpu)
{
	if (m_displayDevice == EDeviceMode::Cpu)
	{
		samplesCpu = 1.f;
		samplesGpu = 0.f;
	}
	else if (m_displayDevice == EDeviceMode::Optix)
	{
		samplesCpu = 0.f;
		samplesGpu = 1.f;
	}
	else
	{
		uint32_t currentCpuSample = m_currentCpuSample;
		const uint32_t cpu = currentCpuSample == 0 ? 1 : currentCpuSample;
		const uint32_t gpu = m_currentOptixSample;
		const float sum = static_cast<float>(cpu + gpu);

		samplesCpu = sum == 0 ? 1.f : cpu / sum;
		samplesGpu = sum == 0 ? 0.f : gpu / sum;
	}
}

NORI_NAMESPACE_END