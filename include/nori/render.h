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
    RenderThread(ImageBlock & block);
    ~RenderThread();

    void renderScene(const std::string & filename);
    void rerenderScene(const std::string& filename);

	inline void renderThreadMain(const std::string &outputName, const std::string &outputNameDenoised,
	                      const std::string &outputNameVariance);

    bool isBusy();
    void stopRendering();

    float getProgress();

    Scene* m_scene = nullptr;
protected:
    
    ImageBlock & m_block;
    std::thread m_render_thread;
    std::atomic<int> m_render_status; // 0: free, 1: busy, 2: interruption, 3: done
    std::atomic<float> m_progress;

};

NORI_NAMESPACE_END

#endif //__NORI_RENDER_H
