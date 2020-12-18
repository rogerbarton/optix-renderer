/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Romain Prévost

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

#if !defined(__NORI_RENDER_H)
#define __NORI_RENDER_H

#include <nori/common.h>
#include <thread>
#include <nori/block.h>
#include <nori/integrator.h>
#include <atomic>

#ifdef NORI_USE_OPTIX
#   include <nori/optix/sutil/CUDAOutputBuffer.h>
#endif

NORI_NAMESPACE_BEGIN

using clock_t = std::chrono::steady_clock;
using time_point_t = std::chrono::time_point<std::chrono::steady_clock>;
template<typename TimePoint>
inline double durationMs(const TimePoint time)
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}

struct EDeviceMode
{
	using Type = int;
	static constexpr int  Cpu                          = 0;
	static constexpr int  Optix                        = 1;
	static constexpr int  Both                         = 2;
	static constexpr int  Size                         = 3;
	static constexpr char *Strings[Size] = {"CPU", "Optix", "CPU + Optix"};
};
using EDeviceMode_t = ERenderLayer::Type;

class RenderThread {

public:
	Scene *m_guiScene    = nullptr;
	Scene *m_renderScene = nullptr;

	RenderThread();
    ~RenderThread();
	void PreGlDestroy();

    void loadScene(const std::string & filename);
    void restartRender();

    bool isBusy();
    void stopRendering();
	float getProgress();
	std::string getRenderTime();
	std::string getFilename() { return sceneFilename; }

#ifdef NORI_USE_IMGUI
	/**
	 * Draws gui specific to rendering. Does not draw the scene gui.
	 */
	void drawRenderGui();
	void drawSceneGui();
#endif

	ERenderLayer_t getVisibleRenderLayer() { return m_visibleRenderLayer; }
	void setVisibleRenderLayer(ERenderLayer_t layer) { m_visibleRenderLayer = layer; }
	ERenderLayer_t getVisibleDevice() { return m_displayDevice; }
	void setVisibleDevice(EDeviceMode_t layer) { m_displayDevice = layer; }

	void initBlocks();
	ImageBlock &getBlock(ERenderLayer_t layer = ERenderLayer::Size);  /// Get the active block
#ifdef NORI_USE_OPTIX
	CUDAOutputBuffer<float4> *getDisplayBlockGpu(ERenderLayer_t layer = ERenderLayer::Size);  /// Get the active optix block
	void updateOptixDisplayBuffers();
#endif

	/**
	 * Returns the samples done by both devices normalized by the total samples.
	 */
	void getDeviceSampleWeights(float &samplesCpu, float &samplesGpu);

protected:
	void startRenderThread();
	void renderThreadMain();
#ifdef NORI_USE_OPTIX
	void renderThreadOptix();
#endif

	enum class ERenderStatus : int {
		Idle      = 0,
		Busy      = 1,
		Interrupt = 2,
		Done      = 3
	};

	/**
	 * Restart render when a change is detected. Otherwise the apply button can be used.
	 * m_preview_mode overrides this.
	 */
	bool        m_autoUpdate      = true;
	bool        m_guiSceneTouched = false;
	bool        m_previewMode     = false;

	EDeviceMode_t m_renderDevice  = EDeviceMode::Both;
	EDeviceMode_t m_displayDevice = EDeviceMode::Both;

	ERenderLayer_t m_visibleRenderLayer = ERenderLayer::Composite;
	ImageBlock m_block;
    ImageBlock m_blockAlbedo;      	/// Albedo feature buffer
    ImageBlock m_blockNormal;		/// Normals feature buffer

    std::thread                m_renderThread;
    std::atomic<ERenderStatus> m_renderStatus = ERenderStatus::Idle;
    std::atomic<float>         m_progress     = 1.f;
	std::atomic<uint32_t>      m_currentCpuSample;
	std::atomic<uint32_t>      m_currentOptixSample;
	time_point_t               m_startTime;
	time_point_t               m_endTime;

#ifdef NORI_USE_OPTIX
	std::thread m_optixThread;
	size_t m_optixBlockSizeBytes = 0;
	float4* m_d_optixRenderBlock = 0;
	float4* m_d_optixRenderBlockAlbedo = 0;
	float4* m_d_optixRenderBlockNormal = 0;
	float4* m_d_optixRenderBlockDenoised = 0;

	CUDAOutputBuffer<float4>* m_optixDisplayBlock;
	CUDAOutputBuffer<float4>* m_optixDisplayBlockAlbedo;
	CUDAOutputBuffer<float4>* m_optixDisplayBlockNormal;
	CUDAOutputBuffer<float4>* m_optixDisplayBlockDenoised;
	std::atomic<bool> m_optixDisplayBlockTouched = false;
	std::atomic<bool> m_optixDisplayBlockDenoisedTouched = false;
	float4* m_d_optixDisplayBlock = 0;
	float4* m_d_optixDisplayBlockAlbedo = 0;
	float4* m_d_optixDisplayBlockNormal = 0;
	float4* m_d_optixDisplayBlockDenoised = 0;
#endif

	std::string sceneFilename;
	std::string outputName;
	std::string outputNameDenoised;
	std::string outputNameVariance;
	};

NORI_NAMESPACE_END

#endif //__NORI_RENDER_H
