#include <nori/transform.h>
#ifndef NORI_USE_NANOGUI
#include <nori/ImguiHelpers.h>
#include <Eigen/Geometry>
#endif

NORI_NAMESPACE_BEGIN

Transform::Transform(const Eigen::Matrix4f &trafo)
    : m_transform(trafo), m_inverse(trafo.inverse()) {}

std::string Transform::toString() const
{
    std::ostringstream oss;
    oss << m_transform.format(Eigen::IOFormat(4, 0, ", ", ";\n", "", "", "[", "]"));
    return oss.str();
}

#ifndef NORI_USE_NANOGUI
bool Transform::getImGuiNodes()
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                               ImGuiTreeNodeFlags_Bullet;

    nori::Vector3f origin = m_transform.col(3).head(3);

    bool ret = false;

    ImGui::AlignTextToFramePadding();
    ImGui::PushID(1);
    ImGui::TreeNodeEx("origin", flags, "Origin");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    ret |= ImGui::DragVector3f("##value", &origin);
    ImGui::PopID();
    ImGui::NextColumn();

    Eigen::Matrix3f rotMat = m_transform.block(0, 0, 3, 3);
    Vector3f eulerAngles = rotMat.eulerAngles(2, 0, 2) * 180.f * INV_PI;

    ImGui::AlignTextToFramePadding();
    ImGui::PushID(2);
    ImGui::TreeNodeEx("eulerAngles", flags, "Euler Angles");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    ret |= ImGui::DragVector3f("##value", &eulerAngles, 0.5f, -360, 360, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopID();
    ImGui::NextColumn();

    if (!ret)
        return ret;

    // convert euler angles + origin back to matrix
    m_transform.col(3).head(3) = origin;

    eulerAngles *= M_PI / 180.f;

    rotMat = Eigen::Quaternionf(
                 Eigen::Quaternionf::Identity() *
                 Eigen::AngleAxisf(eulerAngles.x(), Eigen::Vector3f::UnitZ()) *
                 Eigen::AngleAxisf(eulerAngles.y(), Eigen::Vector3f::UnitX())) *
             Eigen::AngleAxisf(eulerAngles.z(), Eigen::Vector3f::UnitZ())
                 .toRotationMatrix();

    m_transform.block(0, 0, 3, 3) = rotMat;

    // Update inverse as well
    m_inverse = m_transform.inverse();

    return ret;
}
#endif

Transform Transform::operator*(const Transform &t) const
{
    return Transform(m_transform * t.m_transform,
                     t.m_inverse * m_inverse);
}
void Transform::update(const Transform &guiObject)
{
	*this = guiObject;
}

NORI_NAMESPACE_END