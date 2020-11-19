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

NORI_NAMESPACE_BEGIN

template <typename T>
class ConstantTexture : public Texture<T>
{
public:
    ConstantTexture(const PropertyList &props);

    virtual std::string toString() const override;

    virtual T eval(const Point2f &uv) override
    {
        return m_value;
    }
#ifndef NORI_USE_NANOGUI
    virtual const char *getImGuiName() const override { return "Constant"; }
    virtual void getImGuiNodes() override {}
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
#ifndef NORI_USE_NANOGUI
template <>
void ConstantTexture<float>::getImGuiNodes()
{
    Texture::getImGuiNodes();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                               ImGuiTreeNodeFlags_Bullet;

    int id = 1;

    ImGui::AlignTextToFramePadding();

    ImGui::TreeNodeEx("Float", flags, "Float");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);

    ImGui::DragFloat("##value", &m_value, 0.01, 0, 1, "%f%", ImGuiSliderFlags_AlwaysClamp);
    ImGui::NextColumn();
}

template <>
void ConstantTexture<Color3f>::getImGuiNodes()
{
    Texture::getImGuiNodes();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                               ImGuiTreeNodeFlags_Bullet;

    ImGui::AlignTextToFramePadding();

    ImGui::TreeNodeEx("Color", flags, "Color");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);

    ImGui::ColorPicker("##value", &m_value);
    ImGui::NextColumn();
}
#endif

NORI_REGISTER_TEMPLATED_CLASS(ConstantTexture, float, "constant_float")
NORI_REGISTER_TEMPLATED_CLASS(ConstantTexture, Color3f, "constant_color")
NORI_NAMESPACE_END