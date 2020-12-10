#include <nori/emitter.h>

#ifdef NORI_USE_OPTIX
#include <nori/optix/sutil/host_vec_math.h>
#endif

NORI_NAMESPACE_BEGIN

	void Emitter::cloneAndInit(Emitter *clone)
	{
		// Shape already cloned by scene
		// if(m_shape)
		// 	clone->m_shape = static_cast<Shape *>(m_shape->cloneAndInit());
		clone->lightProb = lightProb;
	}

	void Emitter::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const Emitter *>(guiObject);
		lightProb = gui->lightProb;
		// Shape already updated by scene
		// if(m_shape)
		// 	m_shape->update(gui->m_shape);
	}

#ifdef NORI_USE_IMGUI
	bool Emitter::getImGuiNodes()
	{
		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("Radiance", ImGuiLeafNodeFlags, "Radiance");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		touched |= ImGui::DragColor3f("##value", &m_radiance, 0.1f, 0, SLIDER_MAX_FLOAT, "%.3f",
		                              ImGuiSliderFlags_AlwaysClamp);
		ImGui::NextColumn();

		ImGui::PushID(1);
		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("Light Sample Weight", ImGuiLeafNodeFlags, "Light Sample Weight");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		touched |= ImGui::DragFloat("##value", &lightProb, 0.1f, 0.f, SLIDER_MAX_FLOAT);
		ImGui::NextColumn();
		ImGui::PopID();

		// Also set shape dirty in case this is called by the scene directly
		if (m_shape)
			m_shape->touched |= touched;
		return touched;
	}
#endif

#ifdef NORI_USE_OPTIX
	void Emitter::getOptixEmitterData(EmitterData &sbtData)
	{
		sbtData.radiance = make_float3(m_radiance);
	}
#endif

NORI_NAMESPACE_END