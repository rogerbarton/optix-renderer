#include <nori/camera.h>
#include <nori/rfilter.h>

NORI_NAMESPACE_BEGIN
#ifdef NORI_USE_IMGUI
bool Camera::getImGuiNodes()
{
    ImGui::AlignTextToFramePadding();
    ImGui::TreeNodeEx("outputSize", ImGuiLeafNodeFlags, "Output Size");
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
	touched |= ImGui::DragVector2i("##value", &m_outputSize, 1, 0, SLIDER_MAX_INT, "%d", ImGuiSliderFlags_AlwaysClamp);
    ImGui::NextColumn();

    bool node_open = ImGui::TreeNode("Reconstruction Filter");
    ImGui::AlignTextToFramePadding();
    ImGui::NextColumn();
    ImGui::SetNextItemWidth(-1);
    ImGui::Text(m_rfilter->getImGuiName().c_str());
    ImGui::NextColumn();

    if (node_open)
    {
	    touched |= m_rfilter->getImGuiNodes();
        ImGui::TreePop();
    }

    return touched;
}
#endif

NORI_NAMESPACE_END