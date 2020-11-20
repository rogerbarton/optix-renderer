#include <nori/ImguiHelpers.h>

namespace ImGui
{
    IMGUI_IMPL_API bool DragColor3f(const char *label, nori::Color3f *color,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        nori::Vector3f col_val = nori::Vector3f(color->r(), color->g(), color->b());
        float exponent = col_val.maxCoeff();
        col_val /= col_val.maxCoeff();

        bool ret = false;

        ImGui::PushID(42);
        ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.5f);
        ret |= ImGui::ColorPicker3(label, col_val.data(), ImGuiColorEditFlags_DisplayRGB);
        ImGui::PopItemWidth();
        ImGui::PopID();

        ImGui::PushID(1337);
        ret |= ImGui::DragFloat("##value", &exponent, 0.1f, Epsilon, SLIDER_MAX_FLOAT, "Exponent: %.4f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopID();

        // puzzle back the color
        col_val /= col_val.maxCoeff();
        col_val *= exponent;
        color->r() = col_val.x();
        color->g() = col_val.y();
        color->b() = col_val.z();

        return ret;
    }
    IMGUI_IMPL_API bool ColorPicker(const char *label, nori::Color3f *color)
    {

        ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.5f);
        bool ret = ImGui::ColorPicker3(label, color->data(), ImGuiColorEditFlags_DisplayRGB);
        ImGui::PopItemWidth();
        return ret;
    }

    IMGUI_IMPL_API bool DragColor4f(const char *label, nori::Color4f *color,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat4(label, color->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragPoint2f(const char *label, nori::Point2f *p,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat2(label, p->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragPoint3f(const char *label, nori::Point3f *p,
                                    float v, float v_min, float v_max,
                                    const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat3(label, p->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragVector2f(const char *label, nori::Vector2f *vec,
                                     float v, float v_min, float v_max,
                                     const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat2(label, vec->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragVector2i(const char *label, nori::Vector2i *vec,
                                     float v, int v_min, int v_max,
                                     const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragInt2(label, vec->data(), v, v_min, v_max, fmt, flags);
    }

    IMGUI_IMPL_API bool DragVector3f(const char *label, nori::Vector3f *vec,
                                     float v, float v_min, float v_max,
                                     const char *fmt, ImGuiSliderFlags flags)
    {
        return ImGui::DragFloat3(label, vec->data(), v, v_min, v_max, fmt, flags);
    }

} // namespace ImGui