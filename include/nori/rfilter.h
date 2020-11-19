/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

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

#if !defined(__NORI_RFILTER_H)
#define __NORI_RFILTER_H

#include <nori/object.h>

/// Reconstruction filters will be tabulated at this resolution
#define NORI_FILTER_RESOLUTION 32

NORI_NAMESPACE_BEGIN

/**
 * \brief Generic radially symmetric image reconstruction filter
 *
 * When adding radiance-valued samples to the rendered image, Nori
 * first convolves them with a so-called image reconstruction filter.
 *
 * To learn more about reconstruction filters and sampling theory
 * in general, take a look at the excellenent chapter 7 of PBRT,
 * which is freely available at:
 *
 * http://graphics.stanford.edu/~mmp/chapters/pbrt_chapter7.pdf
 */
class ReconstructionFilter : public NoriObject
{
public:
    /// Return the filter radius in fractional pixels
    float getRadius() const { return m_radius; }

    /// Evaluate the filter function
    virtual float eval(float x) const = 0;

    /**
     * \brief Return the type of object (i.e. Mesh/Camera/etc.) 
     * provided by this instance
     * */
    virtual EClassType getClassType() const override { return EReconstructionFilter; }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override { return "Reconstruction Filter"; }
    virtual void getImGuiNodes() override
    {
        ImGui::PushID(EReconstructionFilter);
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                   ImGuiTreeNodeFlags_Bullet;
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("Shape", flags, "Shape");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##value", &m_radius, 1.f, 0.f, SLIDER_MAX_FLOAT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();
    }
    #endif

protected:
    float m_radius;
};

NORI_NAMESPACE_END

#endif /* __NORI_RFILTER_H */
