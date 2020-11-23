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

#include <nori/rfilter.h>

NORI_NAMESPACE_BEGIN

/**
 * Windowed Gaussian filter with configurable extent
 * and standard deviation. Often produces pleasing 
 * results, but may introduce too much blurring.
 */
class GaussianFilter : public ReconstructionFilter
{
public:
    explicit GaussianFilter(const PropertyList &propList)
    {
        /* Half filter size */
        m_radius = propList.getFloat("radius", 2.0f);
        /* Standard deviation of the Gaussian */
        m_stddev = propList.getFloat("stddev", 0.5f);
    }
	NORI_OBJECT_DEFAULT_CLONE(GaussianFilter)
	NORI_OBJECT_DEFAULT_UPDATE(GaussianFilter)

	float eval(float x) const
    {
        float alpha = -1.0f / (2.0f * m_stddev * m_stddev);
        return std::max(0.0f,
                        std::exp(alpha * x * x) -
                            std::exp(alpha * m_radius * m_radius));
    }

    virtual std::string toString() const override
    {
        return tfm::format("GaussianFilter[radius=%f, stddev=%f]", m_radius, m_stddev);
    }
#ifndef NORI_USE_NANOGUI
	NORI_OBJECT_IMGUI_NAME("Gaussian");
    virtual bool getImGuiNodes() override
    {
        bool ret = ReconstructionFilter::getImGuiNodes();

        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("Stddev", ImGuiLeafNodeFlags, "Standard Deviation");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ret |= ImGui::DragFloat("##value", &m_stddev, 0.01f, 0, SLIDER_MAX_FLOAT, "%f%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();

        return ret;
    }
#endif

protected:
    float m_stddev;
};

/**
 * Separable reconstruction filter by Mitchell and Netravali
 * 
 * D. Mitchell, A. Netravali, Reconstruction filters for computer graphics, 
 * Proceedings of SIGGRAPH 88, Computer Graphics 22(4), pp. 221-228, 1988.
 */
class MitchellNetravaliFilter : public ReconstructionFilter
{
public:
    explicit MitchellNetravaliFilter(const PropertyList &propList)
    {
        /* Filter size in pixels */
        m_radius = propList.getFloat("radius", 2.0f);
        /* B parameter from the paper */
        m_B = propList.getFloat("B", 1.0f / 3.0f);
        /* C parameter from the paper */
        m_C = propList.getFloat("C", 1.0f / 3.0f);
    }
	NORI_OBJECT_DEFAULT_CLONE(MitchellNetravaliFilter)
	NORI_OBJECT_DEFAULT_UPDATE(MitchellNetravaliFilter)

	float eval(float x) const
    {
        x = std::abs(2.0f * x / m_radius);
        float x2 = x * x, x3 = x2 * x;

        if (x < 1)
        {
            return 1.0f / 6.0f * ((12 - 9 * m_B - 6 * m_C) * x3 + (-18 + 12 * m_B + 6 * m_C) * x2 + (6 - 2 * m_B));
        }
        else if (x < 2)
        {
            return 1.0f / 6.0f * ((-m_B - 6 * m_C) * x3 + (6 * m_B + 30 * m_C) * x2 + (-12 * m_B - 48 * m_C) * x + (8 * m_B + 24 * m_C));
        }
        else
        {
            return 0.0f;
        }
    }

    virtual std::string toString() const override
    {
        return tfm::format("MitchellNetravaliFilter[radius=%f, B=%f, C=%f]", m_radius, m_B, m_C);
    }
#ifndef NORI_USE_NANOGUI
	NORI_OBJECT_IMGUI_NAME("Mitchell-Netravali");
    virtual bool getImGuiNodes() override
    {
        bool result = ReconstructionFilter::getImGuiNodes();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(1);
        ImGui::TreeNodeEx("m_B", ImGuiLeafNodeFlags, "B");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        result |= ImGui::DragFloat("##value", &m_B, 0.01f, 0, SLIDER_MAX_FLOAT, "%f%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::AlignTextToFramePadding();
        ImGui::PushID(1);
        ImGui::TreeNodeEx("m_C", ImGuiLeafNodeFlags, "C");
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
	    result |= ImGui::DragFloat("##value", &m_C, 0.01f, 0, SLIDER_MAX_FLOAT, "%f%", ImGuiSliderFlags_AlwaysClamp);
        ImGui::NextColumn();
        ImGui::PopID();

	    return result;
    }
#endif
protected:
    float m_B, m_C;
};

/// Tent filter
class TentFilter : public ReconstructionFilter
{
public:
    explicit TentFilter(const PropertyList &)
    {
        m_radius = 1.0f;
    }
	NORI_OBJECT_DEFAULT_CLONE(TentFilter)
	NORI_OBJECT_DEFAULT_UPDATE(TentFilter)

	float eval(float x) const
    {
        return std::max(0.0f, 1.0f - std::abs(x));
    }

    virtual std::string toString() const override
    {
        return "TentFilter[]";
    }
#ifndef NORI_USE_NANOGUI
	NORI_OBJECT_IMGUI_NAME("Tent");
    virtual bool getImGuiNodes() override
    {
        return ReconstructionFilter::getImGuiNodes();
    }
#endif
};

/// Box filter -- fastest, but prone to aliasing
class BoxFilter : public ReconstructionFilter
{
public:
    explicit BoxFilter(const PropertyList &)
    {
        m_radius = 0.5f;
    }
	NORI_OBJECT_DEFAULT_CLONE(BoxFilter)
	NORI_OBJECT_DEFAULT_UPDATE(BoxFilter)

	float eval(float) const
    {
        return 1.0f;
    }

    virtual std::string toString() const override
    {
        return "BoxFilter[]";
    }
#ifndef NORI_USE_NANOGUI
	NORI_OBJECT_IMGUI_NAME("Box");
    virtual bool getImGuiNodes() override
    {
        return ReconstructionFilter::getImGuiNodes();
    }
#endif
};

NORI_REGISTER_CLASS(GaussianFilter, "gaussian");
NORI_REGISTER_CLASS(MitchellNetravaliFilter, "mitchell");
NORI_REGISTER_CLASS(TentFilter, "tent");
NORI_REGISTER_CLASS(BoxFilter, "box");

NORI_NAMESPACE_END
