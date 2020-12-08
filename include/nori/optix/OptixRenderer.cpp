//
// Created by roger on 06/12/2020.
//

#include <nori/optix/OptixRenderer.h>

NORI_NAMESPACE_BEGIN

	OptixRenderer::OptixRenderer(const PropertyList &propList)
	{
		m_enableSpecialization = propList.getBoolean("specialize", true);
		m_samplesPerLaunch     = propList.getInteger("samplesPerLaunch", 16);
	}

	NoriObject *OptixRenderer::cloneAndInit()
	{
		auto clone = new OptixRenderer(*this);
		return clone;
	}

	void OptixRenderer::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const OptixRenderer *>(guiObject);
		if (!gui->touched)return;
		gui->touched = false;
		touchedOptix = true;

		m_samplesPerLaunch     = gui->m_samplesPerLaunch;
		m_enableSpecialization = gui->m_enableSpecialization;
	}

	std::string OptixRenderer::toString() const
	{
		return tfm::format(
				"OptixRenderer[\n"
				"  samplesPerLaunch = %i\n"
				"]",
				m_samplesPerLaunch);
	}

	bool OptixRenderer::getImGuiNodes()
	{
		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("specialize", ImGuiLeafNodeFlags, "Enable Specialization");
		ImGui::SameLine();
		ImGui::HelpMarker("Bind constant values in the launch params. Requires recompilation but gives better performance.");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		touched |= ImGui::Checkbox("##value", &m_enableSpecialization);
		ImGui::NextColumn();

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("samplesPerLaunch", ImGuiLeafNodeFlags, "Samples per Launch");
		ImGui::SameLine();
		ImGui::HelpMarker("Execute multiple samples per launch for better performance, although at slower refresh rates.");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		touched |= ImGui::DragInt("##value", &m_samplesPerLaunch, 1, 1, SLIDER_MAX_INT, "%.3f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::NextColumn();

		return touched;
	}

NORI_NAMESPACE_END
