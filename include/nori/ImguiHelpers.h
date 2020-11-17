#pragma once

#include <imgui/imconfig.h>
#include <imgui/imgui.h>

/**
 * This file contains some useful imgui helpers
 */

/**
 * Easy access to enable or disable ImGui buttons
 * How to use:
 * IMGUI_DISABLE_IF(disable_condition, ImGui-code to set disabled (can be multiline))
 * Note: X can not change during the execution of CODE!
 */
#define IMGUI_DISABLE_IF(CONDITION, CODE)                    \
    if (CONDITION) {                                         \
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);  \
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha,             \
                            ImGui::GetStyle().Alpha * 0.5f); \
    }                                                        \
    CODE;                                                    \
    if (CONDITION) {                                         \
        ImGui::PopItemFlag();                                \
        ImGui::PopStyleVar();                                \
    }