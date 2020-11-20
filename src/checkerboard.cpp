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

#include <nori/object.h>
#include <nori/texture.h>

NORI_NAMESPACE_BEGIN

template <typename T>
class Checkerboard : public Texture<T>
{
public:
  Checkerboard(const PropertyList &props);

  virtual std::string toString() const override;

  virtual T eval(const Point2f &uv) override
  {
    Point2f origin;
    origin.x() = uv.x() / m_scale.x() - m_delta.x();
    origin.y() = uv.y() / m_scale.y() - m_delta.y();

    int x = int(origin.x()) + (origin.x() < 0.f);
    int y = int(origin.y()) + (origin.y() < 0.f);

    // checkerboard check
    if ((x + y) % 2 == 0)
    {
      return m_value1;
    }
    else
    {
      return m_value2;
    }
  }
#ifndef NORI_USE_NANOGUI
  virtual const char *getImGuiName() const override { return "Checkerboard"; }
  virtual bool getImGuiNodes() override { return false; }
#endif
protected:
  T m_value1;
  T m_value2;

  Point2f m_delta;
  Vector2f m_scale;
};

template <>
Checkerboard<float>::Checkerboard(const PropertyList &props)
{
  m_delta = props.getPoint2("delta", Point2f(0));
  m_scale = props.getVector2("scale", Vector2f(1));
  m_value1 = props.getFloat("value1", 0.f);
  m_value2 = props.getFloat("value2", 1.f);
}

template <>
Checkerboard<Color3f>::Checkerboard(const PropertyList &props)
{
  m_delta = props.getPoint2("delta", Point2f(0));
  m_scale = props.getVector2("scale", Vector2f(1));
  m_value1 = props.getColor("value1", Color3f(0));
  m_value2 = props.getColor("value2", Color3f(1));
}

template <>
std::string Checkerboard<float>::toString() const
{
  return tfm::format("Checkerboard[\n"
                     "  delta = %s,\n"
                     "  scale = %s,\n"
                     "  value1 = %f,\n"
                     "  value2 = %f,\n"
                     "]",
                     m_delta.toString(), m_scale.toString(), m_value1,
                     m_value2);
}

template <>
std::string Checkerboard<Color3f>::toString() const
{
  return tfm::format("Checkerboard[\n"
                     "  delta = %s,\n"
                     "  scale = %s,\n"
                     "  tex1 = %s,\n"
                     "  tex2 = %s,\n"
                     "]",
                     m_delta.toString(), m_scale.toString(),
                     m_value1.toString(), m_value2.toString());
}
#ifndef NORI_USE_NANOGUI
template <>
bool Checkerboard<float>::getImGuiNodes()
{
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                             ImGuiTreeNodeFlags_Bullet;
  bool ret = false;
  ImGui::PushID(1);
  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Float 1", flags, "Float 1");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::DragFloat("##value", &m_value1, 0.01, 0, 1, "%f%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::NextColumn();
  ImGui::PopID();

  ImGui::PushID(2);
  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Float 2", flags, "Float 2");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::DragFloat("##value", &m_value2, 0.01, 0, 1, "%f%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::NextColumn();
  ImGui::PopID();

  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Delta", flags, "Delta");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::DragPoint2f("##value", &m_delta, 0.02, 0, 5, "%f%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::NextColumn();

  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Scale", flags, "Scale");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::DragVector2f("##value", &m_scale, 0.02, 0, 5, "%f%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::NextColumn();
  return ret;
}

template <>
bool Checkerboard<Color3f>::getImGuiNodes()
{

  bool ret = Texture::getImGuiNodes();

  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                             ImGuiTreeNodeFlags_Bullet;

  ImGui::PushID(1);
  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Color 1", flags, "Color 1");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::ColorPicker("##value", &m_value1);
  ImGui::NextColumn();
  ImGui::PopID();

  ImGui::PushID(2);
  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Color 2", flags, "Color 2");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::ColorPicker("##value", &m_value2);
  ImGui::NextColumn();
  ImGui::PopID();

  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Delta", flags, "Delta");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::DragPoint2f("##value", &m_delta, 0.02, 0, 5, "%f%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::NextColumn();

  ImGui::AlignTextToFramePadding();
  ImGui::TreeNodeEx("Scale", flags, "Scale");
  ImGui::NextColumn();
  ImGui::SetNextItemWidth(-1);
  ret |= ImGui::DragVector2f("##value", &m_scale, 0.02, 0, 5, "%f%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::NextColumn();
  return ret;
}
#endif

NORI_REGISTER_TEMPLATED_CLASS(Checkerboard, float, "checkerboard_float")
NORI_REGISTER_TEMPLATED_CLASS(Checkerboard, Color3f, "checkerboard_color")
NORI_NAMESPACE_END
