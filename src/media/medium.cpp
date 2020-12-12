//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>
#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN

	Medium::Medium(const PropertyList &propList)
	{
		// Beer's law medium as default, sigma_s = 0
		m_sigma_a_normalized = propList.getColor("sigma_a", 0.5f);
		m_sigma_s_normalized = propList.getColor("sigma_s", 0);
		m_sigma_a_intensity  = propList.getFloat("sigma_a_intensity", 1.f);
		m_sigma_s_intensity  = propList.getFloat("sigma_s_intensity", 1.f);
	}

	void Medium::cloneAndInit(Medium *clone)
	{
		// Use isotropic phase as default phase function
		if (!m_phase)
			m_phase = static_cast<PhaseFunction *>(NoriObjectFactory::createInstance("isophase"));
		clone->m_phase = static_cast<PhaseFunction *>(m_phase->cloneAndInit());

		if (m_emitter)
			clone->m_emitter = static_cast<Emitter *>(m_emitter->cloneAndInit());
	}

	void Medium::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const Medium *>(guiObject);

		m_sigma_a = gui->m_sigma_a;
		m_sigma_s = gui->m_sigma_s;
		m_sigma_t = gui->m_sigma_t;

		m_phase->update(gui->m_phase);
		if (m_emitter)
			m_emitter->update(gui->m_emitter);
	}

	Medium::~Medium()
	{
		delete m_phase;
		delete m_emitter;
	}

	void Medium::addChild(NoriObject *obj)
	{
		switch (obj->getClassType())
		{
			case EPhaseFunction:
				if (m_phase)
					throw NoriException("Medium: tried to register multiple PhaseFunction instances!");
				m_phase = static_cast<PhaseFunction *>(obj);
				break;

			case EEmitter:
				if (m_emitter)
					throw NoriException("Medium: tried to register multiple Emitter instances!");
				m_emitter = static_cast<Emitter *>(obj);
				// Note: emitter shape set by Shape::cloneAndInit
				break;

			default:
				throw NoriException("Medium::addChild(<%s>) is not supported!", classTypeName(obj->getClassType()));
		}
	}

#ifdef NORI_USE_IMGUI
	bool Medium::getImGuiNodes()
	{
		bool newlyTouched = false;

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("sigma_a", ImGuiLeafNodeFlags, "Absorption (sigma_a)");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.75f);
		newlyTouched |= ImGui::ColorPicker3("##sigma_a", reinterpret_cast<float *>(&m_sigma_a_normalized),
		                               ImGuiClampColorEditFlags);
		ImGui::NextColumn();
		ImGui::TreeNodeEx("sigma_a_intensity", ImGuiLeafNodeFlags, "Intensity");
		ImGui::Text("");
		ImGui::NextColumn();
		newlyTouched |= ImGui::DragFloat("##sigma_a_intensity", &m_sigma_a_intensity, 0.001f, 0.f, SLIDER_MAX_FLOAT);
		ImGui::NextColumn();

		ImGui::AlignTextToFramePadding();
		ImGui::TreeNodeEx("sigma_s", ImGuiLeafNodeFlags, "Scattering (sigma_s)");
		ImGui::NextColumn();
		ImGui::SetNextItemWidth(-1);
		ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.75f);
		newlyTouched |= ImGui::ColorPicker3("##sigma_s", reinterpret_cast<float *>(&m_sigma_s_normalized),
		                               ImGuiClampColorEditFlags);
		ImGui::NextColumn();
		ImGui::TreeNodeEx("sigma_s_intensity", ImGuiLeafNodeFlags, "Intensity");
		ImGui::NextColumn();
		newlyTouched |= ImGui::DragFloat("##sigma_s_intensity", &m_sigma_s_intensity, 0.001f, 0.f, SLIDER_MAX_FLOAT);
		ImGui::NextColumn();

		NORI_IMGUI_CHILD_OBJECT(m_phase, "Phase Function")
		NORI_IMGUI_CHILD_OBJECT(m_emitter, "Emitter (Volume)")

		if (newlyTouched)
			updateDerivedProperties();

		touched |= newlyTouched;
		return touched;
	}
	void Medium::updateDerivedProperties()
	{
		m_sigma_a = m_sigma_a_normalized * m_sigma_a_intensity;
		m_sigma_s = m_sigma_s_normalized * m_sigma_s_intensity;
		m_sigma_t = m_sigma_a + m_sigma_s;
	}
#endif

#ifdef NORI_USE_OPTIX
	void Medium::getOptixMediumData(MediumData &sbtData)
	{

	}
#endif

NORI_NAMESPACE_END