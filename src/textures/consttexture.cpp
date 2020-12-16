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

#include <nori/texture.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

NORI_NAMESPACE_BEGIN

template <typename T>
class ConstantTexture : public Texture<T>
{
public:
    explicit ConstantTexture(const PropertyList &props);
    NORI_OBJECT_DEFAULT_CLONE(ConstantTexture<T>);
    NORI_OBJECT_DEFAULT_UPDATE(ConstantTexture<T>);

    virtual std::string toString() const override;

    virtual T eval(const Point2f &uv) override
    {
        return m_value;
    }
#ifdef NORI_USE_IMGUI
	NORI_OBJECT_IMGUI_NAME("Constant");
    virtual bool getImGuiNodes() override { return false; }
#endif

#ifdef NORI_USE_OPTIX
	void getOptixTexture(float3 &constValue, cudaTextureObject_t &texValue);
#endif
protected:
    T m_value;
};

template <>
ConstantTexture<float>::ConstantTexture(const PropertyList &props)
{
    m_value = props.getFloat("value", 0.f);
}
template <>
ConstantTexture<Color3f>::ConstantTexture(const PropertyList &props)
{
    m_value = props.getColor("value", Color3f(0.f));
}

template <>
std::string ConstantTexture<float>::toString() const
{
    return tfm::format(
        "ConstantTexture[ %f ]",
        m_value);
}

template <>
std::string ConstantTexture<Color3f>::toString() const
{
    return tfm::format(
        "ConstantTexture[ %s ]",
        m_value.toString());
}
#ifdef NORI_USE_IMGUI
template <>
bool ConstantTexture<float>::getImGuiNodes()
{
    touched |= Texture::getImGuiNodes();

    ImGui::AlignTextToFramePadding();

    ImGui::TreeNodeEx("Float", ImGuiLeafNodeFlags, "Float");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);

	touched |= ImGui::DragFloat("##value", &m_value, 0.01f, 0, 1, "%f%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::NextColumn();
    return touched;
}

template <>
bool ConstantTexture<Color3f>::getImGuiNodes()
{
	touched |= Texture::getImGuiNodes();

    ImGui::AlignTextToFramePadding();

    ImGui::TreeNodeEx("Color", ImGuiLeafNodeFlags, "Color");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);

	touched |= ImGui::ColorPicker("##value", &m_value);
    ImGui::NextColumn();
    return touched;
}
#endif

#ifdef NORI_USE_OPTIX
	template<>
	void ConstantTexture<float>::getOptixTexture(float3 &constValue, cudaTextureObject_t &texValue)
	{
		texValue = 0;
		constValue = make_float3(m_value);
	}

	template<>
	void ConstantTexture<Color3f>::getOptixTexture(float3 &constValue, cudaTextureObject_t &texValue)
	{
		texValue = 0;
		constValue = make_float3(m_value);
	}
#endif

NORI_REGISTER_TEMPLATED_CLASS(ConstantTexture, float, "constant_float")
NORI_REGISTER_TEMPLATED_CLASS(ConstantTexture, Color3f, "constant_color")
NORI_NAMESPACE_END