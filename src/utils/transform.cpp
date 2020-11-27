#include <nori/transform.h>
#ifdef NORI_USE_IMGUI
#include <nori/ImguiHelpers.h>
#endif
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

Transform::Transform(const Eigen::Matrix4f &trafo)
    : m_transform(trafo), m_inverse(trafo.inverse()) {}

std::string Transform::toString() const
{
    std::ostringstream oss;
    oss << m_transform.format(Eigen::IOFormat(4, 0, ", ", ";\n", "", "", "[", "]"));
    return oss.str();
}

#ifdef NORI_USE_IMGUI
bool Transform::getImGuiNodes()
{
    nori::Vector3f origin = m_transform.col(3).head(3);

    ImGui::AlignTextToFramePadding();
    ImGui::PushID(1);
    ImGui::TreeNodeEx("origin", ImGuiLeafNodeFlags, "Origin");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    touched |= ImGui::DragVector3f("##value", &origin, 0.01f);
    ImGui::PopID();
    ImGui::NextColumn();

    Eigen::Matrix3f rotScale = m_transform.block(0, 0, 3, 3);
    Vector3f scale = rotScale.colwise().norm();
    Eigen::Matrix3f rotMat = rotScale.colwise().normalized();
    Vector3f eulerAngles = rotMat.eulerAngles(2, 0, 2) * 180.f * INV_PI;

    ImGui::AlignTextToFramePadding();
    ImGui::PushID(2);
    ImGui::TreeNodeEx("eulerAngles", ImGuiLeafNodeFlags, "Euler Angles");
    ImGui::SameLine();
    ImGui::HelpMarker("Use with caution!");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    bool rotMatTouched = ImGui::DragVector3f("##value", &eulerAngles, 0.5f, -360, 360, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopID();
    ImGui::NextColumn();
    touched |= rotMatTouched;

    ImGui::AlignTextToFramePadding();
    ImGui::PushID(3);
    ImGui::TreeNodeEx("scale", ImGuiLeafNodeFlags, "Scale");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    touched |= ImGui::DragVector3f("##value", &scale, 0.01f, 0.01f);
    ImGui::PopID();
    ImGui::NextColumn();

    ImGui::NextColumn();
    ImGui::NextColumn();
    if (ImGui::Button("Reset"))
    {
        touched = true;
        clear();
    }

    if (!touched)
        return touched;

    // Apply translation and scaling
    if (rotMatTouched)
    {
        eulerAngles *= M_PI / 180.f;
        rotMat = Eigen::Quaternionf(
                     Eigen::Quaternionf::Identity() *
                     Eigen::AngleAxisf(eulerAngles.x(), Eigen::Vector3f::UnitZ()) *
                     Eigen::AngleAxisf(eulerAngles.y(), Eigen::Vector3f::UnitX())) *
                 Eigen::AngleAxisf(eulerAngles.z(), Eigen::Vector3f::UnitZ())
                     .toRotationMatrix();
    }

    m_transform.col(3).head(3) = origin;
    m_transform.block(0, 0, 3, 3) = Eigen::Scaling(scale) * rotMat;

    // Update inverse as well
    m_inverse = m_transform.inverse();

    return touched;
}
#endif

Transform Transform::operator*(const Transform &t) const
{
    return Transform(m_transform * t.m_transform,
                     t.m_inverse * m_inverse);
}
void Transform::update(const Transform &guiObject)
{
    if (!guiObject.touched)
        return;
    guiObject.touched = false;
    *this = guiObject;
}

NORI_NAMESPACE_END