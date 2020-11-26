#pragma once

#ifdef NORI_USE_IMGUI
#include <imgui/imgui.h>
#include <nori/vector.h>
#include <nori/color.h>

/**
 * This file contains some useful imgui helpers
 */

namespace nori
{
	const static ImGuiTreeNodeFlags ImGuiLeafNodeFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen |
	                           ImGuiTreeNodeFlags_Bullet;

#define SLIDER_MAX_INT 1000000
#define SLIDER_MAX_FLOAT 1000000.f
} // namespace nori

namespace ImGui
{
    IMGUI_API bool DragColor3f(const char *label, nori::Color3f *color,
                               float v = 1.f, float v_min = 0.f, float v_max = 0.f,
                               const char *fmt = "%.3f", ImGuiSliderFlags flags = 0);
    IMGUI_API bool ColorPicker(const char *label, nori::Color3f *color);
    IMGUI_API bool DragColor4f(const char *label, nori::Color4f *color,
                               float v = 1.f, float v_min = 0.f, float v_max = 0.f,
                               const char *fmt = "%.3f", ImGuiSliderFlags flags = 0);
    IMGUI_API bool DragPoint2f(const char *label, nori::Point2f *p,
                               float v = 1.f, float v_min = 0.f, float v_max = 0.f,
                               const char *fmt = "%.3f", ImGuiSliderFlags flags = 0);
    IMGUI_API bool DragPoint3f(const char *label, nori::Point3f *p,
                               float v = 1.f, float v_min = 0.f, float v_max = 0.f,
                               const char *fmt = "%.3f", ImGuiSliderFlags flags = 0);
    IMGUI_API bool DragVector2f(const char *label, nori::Vector2f *vec,
                                float v = 1.f, float v_min = 0.f, float v_max = 0.f,
                                const char *fmt = "%.3f", ImGuiSliderFlags flags = 0);
    IMGUI_API bool DragVector2i(const char *label, nori::Vector2i *vec,
                                float v = 1.f, int v_min = 0.f, int v_max = 0.f,
                                const char *fmt = "%d", ImGuiSliderFlags flags = 0);
    IMGUI_API bool DragVector3f(const char *label, nori::Vector3f *vec,
                                float v = 1.f, float v_min = 0.f, float v_max = 0.f,
                                const char *fmt = "%.3f", ImGuiSliderFlags flags = 0);

	// Helper to display a little (?) mark which shows a tooltip when hovered.
	// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
	IMGUI_API void HelpMarker(const char* desc);
} // namespace ImGui

#endif