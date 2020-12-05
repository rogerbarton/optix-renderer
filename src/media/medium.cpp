//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>
#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN

	void Medium::cloneAndInit(Medium *clone)
	{
		// Use isotropic phase as default phase function
		if (!m_phase)
			m_phase = static_cast<PhaseFunction *>(NoriObjectFactory::createInstance("isophase", PropertyList()));
		clone->m_phase = static_cast<PhaseFunction *>(m_phase->cloneAndInit());

		if (m_emitter)
			clone->m_emitter = static_cast<Emitter *>(m_emitter->cloneAndInit());
	}

	void Medium::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const Medium *>(guiObject);
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
		ImGui::PushID(EMedium);

		if (m_phase)
		{
			bool nodeOpen = ImGui::TreeNode("Phase Function");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();

			ImGui::Text(m_phase->getImGuiName().c_str());
			ImGui::NextColumn();
			if (nodeOpen)
			{
				touched |= m_phase->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		if (m_emitter)
		{
			bool nodeOpen = ImGui::TreeNode("Emitter (Volume)");
			ImGui::NextColumn();
			ImGui::AlignTextToFramePadding();

			ImGui::Text(m_emitter->getImGuiName().c_str());
			ImGui::NextColumn();
			if (nodeOpen)
			{
				touched |= m_emitter->getImGuiNodes();
				ImGui::TreePop();
			}
		}

		ImGui::PopID();
		return touched;
	}
#endif

NORI_NAMESPACE_END