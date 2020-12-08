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

class RenderThread {

public:
	RenderThread();
    ~RenderThread();

    void loadScene(const std::string & filename);
    void restartRender();

	void renderThreadMain();
#ifdef NORI_USE_OPTIX
	void renderThreadOptix();
	CUDAOutputBuffer<float4> m_optixBlock = {CUDAOutputBufferType::GL_INTEROP, 1, 1};
#endif

    bool isBusy();
    void stopRendering();
	float getProgress() { return isBusy() ? (float) m_progress : 1.f; }
	std::string getRenderTime();
	std::string getFilename() { return sceneFilename; }

#ifdef NORI_USE_IMGUI
	/**
	 * Draws gui specific to rendering. Does not draw the scene gui.
	 */
	void drawRenderGui();
	void drawSceneGui();
#endif

	Scene       *m_guiScene       = nullptr;
	Scene       *m_renderScene    = nullptr;
	/**
	 * Restart render when a change is detected. Otherwise the apply button can be used.
	 * m_preview_mode overrides this.
	 */
	bool        m_autoUpdate      = true;
	bool        m_guiSceneTouched = false;
	bool        m_previewMode     = false;

	static constexpr int EDeviceModeSize = 3;
	const char* m_deviceModeStrings[EDeviceModeSize] = {"CPU", "Optix", "CPU + Optix"};
	enum class EDeviceMode : int {
		Cpu   = 0,
		Optix = 1,
		Both  = 2,
	};
	EDeviceMode m_deviceMode                         = EDeviceMode::Both;

	ERenderLayer_t m_visibleRenderLayer = ERenderLayer::Composite;
	ImageBlock& getCurrentBlock();                      							/// Get the active block
	ImageBlock& getBlock(ERenderLayer_t renderLayer = ERenderLayer::Composite);  	/// Get a specific block
protected:

	enum class ERenderStatus : int {
		Idle      = 0,
		Busy      = 1,
		Interrupt = 2,
		Done      = 3
	};

    ImageBlock m_block;
    ImageBlock m_blockNormal;		/// Normals feature buffer
    ImageBlock m_blockAlbedo;      	/// Albedo feature buffer

    std::thread                m_renderThread;
    std::atomic<ERenderStatus> m_renderStatus = ERenderStatus::Idle;
    std::atomic<float>         m_progress     = 1.f;
	std::atomic<uint32_t>      m_currentCpuSample;
	std::atomic<uint32_t>      m_currentOptixSample;
	time_point_t               m_startTime;
	time_point_t               m_endTime;
#ifdef NORI_USE_OPTIX
	std::thread m_optixThread;
#endif

	std::string sceneFilename;
	std::string outputName;
	std::string outputNameDenoised;
	std::string outputNameVariance;
	};

NORI_NAMESPACE_END

#endif //__NORI_RENDER_H
