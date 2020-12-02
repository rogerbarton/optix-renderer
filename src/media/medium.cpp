//
// Created by roger on 01/12/2020.
//

#include <nori/medium.h>

NORI_NAMESPACE_BEGIN

	void Medium::cloneAndInit(Medium *clone)
	{
		// Use isotropic phase as default phase function
		if (!m_phase)
			m_phase = static_cast<PhaseFunction *>(NoriObjectFactory::createInstance("isophase", PropertyList()));
		clone->m_phase = static_cast<PhaseFunction *>(m_phase->cloneAndInit());
	}

	void Medium::update(const NoriObject *guiObject)
	{
		const auto *gui = static_cast<const Medium *>(guiObject);
		m_phase->update(gui->m_phase);
	}

	Medium::~Medium () {
		delete m_phase;
	}

	void Medium::addChild(NoriObject *obj)
	{
		switch (obj->getClassType())
		{
			case EBSDF:
				if (m_phase)
					throw NoriException("Medium: tried to register multiple PhaseFunction instances!");
				m_phase = static_cast<PhaseFunction *>(obj);
				break;

			default:
				throw NoriException("Medium::addChild(<%s>) is not supported!", classTypeName(obj->getClassType()));
		}
	}
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

		ImGui::PopID();
		return touched;
	}

NORI_NAMESPACE_END