#include <nori/camera.h>
#include <nori/rfilter.h>

NORI_NAMESPACE_BEGIN
#ifndef NORI_USE_NANOGUI
void Camera::getImGuiNodes()
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                               ImGuiTreeNodeFlags_Bullet;

    ImGui::AlignTextToFramePadding();
    ImGui::TreeNodeEx("outputSize", flags, "Output Size");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    ImGui::DragVector2i("Output Size", &m_outputSize, 1, 0, SLIDER_MAX_INT, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::NextColumn();

    bool node_open = ImGui::TreeNode("Reconstruction Filter");
    ImGui::AlignTextToFramePadding();
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    ImGui::Text(m_rfilter->getImGuiName());
    ImGui::NextColumn();

    if (node_open)
    {
        m_rfilter->getImGuiNodes();
        ImGui::TreePop();
    }
}
#endif

NORI_NAMESPACE_END