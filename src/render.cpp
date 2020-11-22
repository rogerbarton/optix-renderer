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

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <filesystem/resolver.h>
#include <fstream>

NORI_NAMESPACE_BEGIN

	static void renderBlock(const Scene *scene, Sampler *sampler, ImageBlock &block, const Histogram &histogram);

	RenderThread::~RenderThread()
	{
		stopRendering();
		delete m_guiScene;
		delete m_renderScene;
	}

	bool RenderThread::isBusy()
	{
		if (m_render_status == ERenderStatus::Done)
		{
			m_render_thread.join();
			m_render_status = ERenderStatus::Idle;
		}
		return m_render_status != ERenderStatus::Idle;
	}

	void RenderThread::stopRendering()
	{
		if (isBusy())
		{
			cout << "Requesting interruption of the current rendering" << endl;
			m_render_status = ERenderStatus::Interrupt;
			m_render_thread.join();
			m_render_status = ERenderStatus::Idle;
			cout << "Rendering successfully aborted" << endl;
		}
	}

	void RenderThread::loadScene(const std::string &filename)
	{
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

		// New scene accepted
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
		m_guiScene    = dynamic_cast<Scene *>(root);
		m_renderScene = dynamic_cast<Scene *>(m_guiScene->cloneAndInit());

		// Determine the filename of the output bitmap
		outputName = sceneFilename;
		size_t      lastdot    = outputName.find_last_of(".");
		if (lastdot != std::string::npos)
			outputName.erase(lastdot, std::string::npos);

		outputNameDenoised = outputName + "_denoised.exr";
		outputNameVariance = outputName + "_variance.dat";
		outputName += ".exr";

		// Start the actual thread
		m_render_status = ERenderStatus::Busy;
		m_render_thread = std::thread([this] { renderThreadMain(); });
	}

	void RenderThread::restartRender()
	{
		if (!m_guiScene)
			return;

		stopRendering();

		m_renderScene->update(m_guiScene); // reinitialize it

		m_render_status = ERenderStatus::Busy;
		m_render_thread = std::thread([this] { renderThreadMain(); });
	}

	void RenderThread::renderThreadMain()
	{
		// call initialize here? call inverse recursively, initialize children first.
		m_renderScene->update(nullptr);

		/* Allocate memory for the entire output image and clear it */
		const Camera *camera    = m_renderScene->getCamera();
		m_block.init(camera->getOutputSize(), camera->getReconstructionFilter());
		m_block.clear();

		cout << "Rendering .. " << std::flush;

		/* Create a block generator (i.e. a work scheduler) */
		Vector2i     outputSize = camera->getOutputSize();
		BlockGenerator blockGenerator(outputSize, NORI_BLOCK_SIZE);

		Timer timer;

		m_renderScene->getIntegrator()->preprocess(m_renderScene);

		const auto numSamples = m_renderScene->getSampler()->getSampleCount();
		const auto numBlocks  = blockGenerator.getBlockCount();

		tbb::concurrent_vector<std::unique_ptr<Sampler>> samplers;
		samplers.resize(numBlocks);

		for (uint32_t k = 0; k < numSamples; ++k)
		{
			m_progress = k / float(numSamples);
			if (m_render_status == ERenderStatus::Interrupt)
				break;

			tbb::blocked_range<int> range(0, numBlocks);

			// calculate variance here
			Eigen::MatrixXf variance(m_block.rows(), m_block.cols());
			m_renderScene->getSampler()->setSampleRound(k);

			Histogram histogram;

			auto map = [&](const tbb::blocked_range<int> &range) {
				// Allocate memory for a small image block to be rendered by the current thread
				ImageBlock block(Vector2i(NORI_BLOCK_SIZE),
				                 camera->getReconstructionFilter());

				for (int i = range.begin(); i < range.end(); ++i)
				{
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

					samplers.at(blockId)->setSampleRound(k);

					// Render all contained pixels
					renderBlock(m_renderScene, samplers.at(blockId).get(), block, histogram);

					// The image block has been processed. Now add it to the "big" block that represents the entire image
					m_block.put(block); // save to master block
				}
			};

			if (m_renderScene->getSampler()->computeVariance())
			{

				// compute variance of the whole image, once for all samplers

				tbb::blocked_range<int> im_range(0, m_block.rows());

				auto im_map = [&](const tbb::blocked_range<int> &range) {
					for (int i = range.begin(); i < range.end(); i++)
					{
						for (int j = 0; j < m_block.cols(); j++)
						{
							Color3f  curr(0.f);
							Color3f  middleCol(0.f);
							int      n = 0; // how many points are actually valid
							for (int k = 0; k < 3; k++)
							{
								for (int l = 0; l < 3; l++)
								{
									int k_ = clamp(i + k, 0, m_block.rows() - 1);
									int l_ = clamp(j + l, 0, m_block.cols() - 1);
									if (k_ != i + k || l_ != j + l)
										continue; // we must skip this one

									n++;

									if (k == 1 && l == 1)
									{
										middleCol = m_block(k_, l_).divideByFilterWeight();
									}
									curr -= m_block(k_, l_).divideByFilterWeight();
									n++;
								}
							}
							curr       = curr / n + middleCol;

							curr = curr.cwiseAbs(); // take abs because we might have a negative color (because of stencil)
							if (curr.isValid())
								variance(i, j) = Eigen::Vector3f(curr).norm();
							else
							{
								variance(i, j) = 0.f;
								std::cout << "Got invalid variance at pixel " << i << "/"
								          << j << ": " << curr.transpose() << std::endl;
							}
						}
					}
				};

				// compute variance in parallel
				tbb::parallel_for(im_range, im_map);

				// normalize variance

				/*
				// and remove top 5%
				float sum = variance.sum() * 0.95f;

				if (variance.sum() > Epsilon)
					variance = variance / sum;

				// clamp entries to 1 top
				for(int i = 0; i < variance.size(); i++) {
					variance(i) = clamp(variance(i), 0.f, 1.f);
				}*/

				// normalize to have prob 1 overall
				if (variance.sum() > Epsilon)
					variance = variance / variance.sum();

				// build cumulative variance matrix
				for (int i = 0; i < variance.rows(); i++)
				{
					for (int j = 0; j < variance.cols(); j++)
					{
						histogram.add_element(i, j, variance(i, j));
					}
				}

				if (k == numSamples - 1 && m_renderScene->getSampler()->computeVariance())
				{
					// write variance to disk
					std::ofstream var_out(outputNameVariance);

					std::cout << std::endl
					          << "Writing variance to " << outputNameVariance << std::endl;

					var_out << variance << std::endl;
					var_out.close();
				}

				ImageBlock block(m_block.getSize(),
				                 camera->getReconstructionFilter());
				m_renderScene->getSampler()->setSampleRound(k);
				// Render all contained pixels
				renderBlock(m_renderScene, m_renderScene->getSampler(), block, histogram);
				m_block.put(block);
			}
			else
			{
				tbb::parallel_for(range, map);
			}

			blockGenerator.reset();
		}

		cout << "done. (took " << timer.elapsedString() << ")" << endl;

		if (m_renderScene->isPreviewMode())
		{
			// stop the rendering here, don't save
			m_render_status = ERenderStatus::Done;
			return;
		}

		/* Now turn the rendered image block into
		   a properly normalized bitmap */
		m_block.lock();
		std::unique_ptr<Bitmap> bitmap(m_block.toBitmap());
		m_block.unlock();

		/* apply the denoiser */
		if (m_renderScene->getDenoiser())
		{
			std::unique_ptr<Bitmap> bitmap_denoised(m_renderScene->getDenoiser()->denoise(bitmap.get()));
			bitmap_denoised->save(outputNameDenoised);
		}

		/* Save using the OpenEXR format */
		bitmap->save(outputName);

		//delete m_scene;
		//m_scene = nullptr;

		m_render_status = ERenderStatus::Done;
	}

	static void renderBlock(const Scene *scene, Sampler *sampler, ImageBlock &block, const Histogram &histogram)
	{
		const Camera     *camera     = scene->getCamera();
		const Integrator *integrator = scene->getIntegrator();

		Point2i offset = block.getOffset();

		/* Clear the block contents */
		block.clear();

		/* For each pixel and pixel sample sample */
		std::vector<std::pair<int, int>> sampleIndices = sampler->getSampleIndices(block, histogram);

		// add a tbb range to the index
		tbb::blocked_range<int> range(0, sampleIndices.size());

		auto map = [&](tbb::blocked_range<int> &range) {
			for (int i = range.begin(); i < range.end(); i++)
			{
				std::pair<int, int> pair           = sampleIndices[i];
				int                 x              = pair.first;
				int                 y              = pair.second;
				Point2f             pixelSample    = Point2f((float) (x + offset.x()), (float) (y + offset.y())) +
				                                     sampler->next2D();
				Point2f             apertureSample = sampler->next2D();

				/* Sample a ray from the camera */
				Ray3f   ray;
				Color3f value                      = camera->sampleRay(ray, pixelSample, apertureSample);

				/* Compute the incident radiance */
				value *= integrator->Li(scene, sampler, ray);

				/* Store in the image block */
				block.put(pixelSample, value);
			}
		};

		tbb::parallel_for(range,
		                  map); // execute this in parallel (applicable if we render the whole block at once (adaptive sampling))
	}

	void RenderThread::drawGui()
	{
		// Start columns
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 2));
		ImGui::Columns(2);

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("fileName", ImGuiLeafNodeFlags, "Filename");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::Text(filesystem::path(sceneFilename).filename().c_str());
		ImGui::NextColumn();

		guiSceneDirty |= m_guiScene->getImGuiNodes();

		// end columns
		ImGui::Columns(1);
		ImGui::Separator();
		ImGui::PopStyleVar();

		if(guiSceneDirty)
		{
			restartRender();
		}
	}


NORI_NAMESPACE_END