#include <nori/camera.h>
#include <nori/rfilter.h>

NORI_NAMESPACE_BEGIN
#ifndef NORI_USE_NANOGUI
bool Camera::getImGuiNodes()
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                               ImGuiTreeNodeFlags_Bullet;
    bool ret = false;
    ImGui::AlignTextToFramePadding();
    ImGui::TreeNodeEx("outputSize", flags, "Output Size");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    ret |= ImGui::DragVector2i("##value", &m_outputSize, 1, 0, SLIDER_MAX_INT, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::NextColumn();

    bool node_open = ImGui::TreeNode("Reconstruction Filter");
    ImGui::AlignTextToFramePadding();
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    ImGui::Text(m_rfilter->getImGuiName());
    ImGui::NextColumn();

    if (node_open)
    {
        ret |= m_rfilter->getImGuiNodes();
        ImGui::TreePop();
    }

    return ret;
}
#endif

NORI_NAMESPACE_END