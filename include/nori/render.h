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
#include <atomic>

NORI_NAMESPACE_BEGIN

class RenderThread {

public:
	RenderThread(ImageBlock &block) : m_block(block) {}
    ~RenderThread();

    void loadScene(const std::string & filename);
    void restartRender();

	inline void renderThreadMain();

    bool isBusy();
    void stopRendering();
	float getProgress() { return isBusy() ? (float) m_progress : 1.f; }

    void drawGui();

	Scene* m_guiScene = nullptr;
    Scene* m_renderScene = nullptr;
protected:

	enum class ERenderStatus : int {
		Idle      = 0,
		Busy      = 1,
		Interrupt = 2,
		Done      = 3
	};

    ImageBlock & m_block;
    std::thread m_render_thread;
    std::atomic<ERenderStatus> m_render_status = ERenderStatus::Idle;
    std::atomic<float> m_progress = 1.f;

    // Update flags
    enum class ERenderThreadUpdateFlags : int{
    	RestartRender = 0,
    	ReloadScene = 1
    };

    ERenderThreadUpdateFlags updateFlags;
    bool guiSceneDirty = false;

	std::string sceneFilename;
	std::string outputName;
	std::string outputNameDenoised;
	std::string outputNameVariance;

	void initializeFromScene();
};

NORI_NAMESPACE_END

#endif //__NORI_RENDER_H
