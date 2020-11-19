#include <nori/ImguiHelpers.h>

namespace ImGui
{
    IMGUI_IMPL_API bool DragColor3f(const char *label, nori::Color3f *color,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat3("##value", color->data(), v, v_min, v_max, fmt, flags);
    }
    IMGUI_IMPL_API bool DragColor4f(const char *label, nori::Color4f *color,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat4("##value", color->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragPoint2f(const char *label, nori::Point2f *p,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat2("##value", p->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragPoint3f(const char *label, nori::Point3f *p,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat3("##value", p->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragVector2f(const char *label, nori::Vector2f *vec,
                                     float v, float v_min, float v_max,
                                     const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat2("##value", vec->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragVector2i(const char *label, nori::Vector2i *vec,
                                     float v, int v_min, int v_max,
                                     const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragInt2("##value", vec->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragVector3f(const char *label, nori::Vector3f *vec,
                                     float v, float v_min, float v_max,
                                     const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat3("##value", vec->data(), v, v_min, v_max, fmt, flags);
    }

} // namespace ImGui