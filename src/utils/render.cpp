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
#include <filesystem/resolver.h>
#include <fstream>

NORI_NAMESPACE_BEGIN

static void renderBlock(const Scene *const scene, Integrator *const integrator, Sampler *const sampler,
						ImageBlock &block);

RenderThread::~RenderThread()
{
	stopRendering();
	delete m_guiScene;
	delete m_renderScene;
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

	// Start the actual thread
	m_renderStatus = ERenderStatus::Busy;
	m_renderThread = std::thread([this] { renderThreadMain(); });
}

void RenderThread::restartRender()
{
	m_guiSceneTouched = false;
	if (!m_guiScene)
		return;

	stopRendering();

	m_renderScene->update(m_guiScene);

	m_renderStatus = ERenderStatus::Busy;
	m_renderThread = std::thread([this] { renderThreadMain(); });
}

void RenderThread::renderThreadMain()
{
	m_startTime = clock_t::now();
	m_endTime = time_point_t::min();
	cout << "Rendering .. " << std::flush;

	/* Allocate memory for the entire output image and clear it */
	const Camera *camera = m_renderScene->getCamera();
	const Vector2i outputSize = camera->getOutputSize();
	m_block.lock();
	m_block.init(outputSize, camera->getReconstructionFilter());
	m_block.clear();
	m_block.unlock();


#ifdef NORI_USE_OPTIX
	if(!m_previewMode)
		m_optixThread = std::thread([this] { renderThreadOptix(); });
#endif

	/* Create a block generator (i.e. a work scheduler) */
	const int blockSize = m_renderScene->getSampler()->isAdaptive() ? NORI_BLOCK_SIZE_ADAPTIVE : NORI_BLOCK_SIZE;
	BlockGenerator blockGenerator(outputSize, blockSize);

	Timer timer;

	Integrator *const integrator = m_renderScene->getIntegrator(m_previewMode);
	integrator->preprocess(m_renderScene);

	auto numSamples = m_previewMode ? 1 : m_renderScene->getSampler()->getSampleCount();
	const auto numBlocks = blockGenerator.getBlockCount();

	tbb::concurrent_vector<std::unique_ptr<Sampler>> samplers;
	samplers.resize(numBlocks);

	std::cout << std::endl;
	for (uint32_t k = 0; k < numSamples; ++k)
	{
		m_progress = k / float(numSamples);
		if (m_renderStatus == ERenderStatus::Interrupt)
			break;

		tbb::blocked_range<int> range(0, numBlocks);

		m_renderScene->getSampler()->setSampleRound(k);

		auto map = [&](const tbb::blocked_range<int> &range) {
			// Allocate memory for a small image block to be rendered by the current thread
			ImageBlock block(Vector2i(blockSize), camera->getReconstructionFilter());

			for (int i = range.begin(); i < range.end(); ++i)
			{
				// i = index of current rendering block
				if (m_renderStatus == ERenderStatus::Interrupt)
					break;

				// Request an image block from the block generator
				blockGenerator.next(block);

				// Get block id to continue using the same sampler
				auto blockId = block.getBlockId();
				if (k == 0)
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
						renderBlock(m_renderScene, integrator, samplers.at(blockId).get(), block);
						m_block.put(block); // save to master block / live view of rendering
						samplers.at(blockId)->setSampleRound(count++);
					}
				}
				else
				{
					// for uniform sampling, simply call render on the block
					renderBlock(m_renderScene, integrator, samplers.at(blockId).get(), block);
					m_block.put(block); // save to master block
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
		BlockGenerator blockGenerator(outputSize, blockSize);
		ReconstructionFilter *rf = static_cast<ReconstructionFilter *>(NoriObjectFactory::createInstance("box"));
		ImageBlock currVarBlock(Vector2i(blockSize), rf);

		ImageBlock fullVarianceMatrix(outputSize, rf);
		fullVarianceMatrix.clear();
		const int blocks = blockGenerator.getBlockCount();

		for (int i = 0; i < blocks; i++)
		{
			blockGenerator.next(currVarBlock);
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
						ImageBlock &block)
{
	const Camera *camera = scene->getCamera();

	Point2i offset = block.getOffset();

	/* For each pixel and pixel sample sample */
	std::vector<std::pair<int, int>> sampleIndices = sampler->getSampleIndices(block);

	/* Clear the block contents */
	block.clear();

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
		value *= integrator->Li(scene, sampler, ray);

		/* Store in the image block */
		block.put(pixelSample, value);
	}
}

#ifdef NORI_USE_OPTIX
void RenderThread::renderThreadOptix()
{
	Vector2i imageDim = m_renderScene->getCamera()->getOutputSize();
	const uint32_t width = imageDim.x();
	const uint32_t height = imageDim.y();
	m_optixBlock.resize(width, height);

	uint32_t numSamples = m_renderScene->getSampler()->getSampleCount();
	for (uint32_t k = 0; k < numSamples; ++k)
	{
		if (m_renderStatus == ERenderStatus::Interrupt)
			break;
		m_renderScene->m_optixRenderer->renderOptixState(m_optixBlock);
	}
}
#endif

#ifdef NORI_USE_IMGUI
void RenderThread::drawRenderGui()
{
	m_guiSceneTouched |= ImGui::Checkbox("Preview", &m_previewMode);

	ImGui::Checkbox("Auto-update", &m_autoUpdate);
	if (m_guiSceneTouched)
	{
		ImGui::SameLine();
		if (ImGui::Button("Apply Changes"))
			restartRender();
	}
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

NORI_NAMESPACE_END